//! wgpu compute backend (Vulkan / Metal / DX12 / GL via wgpu, single-precision).
//!
//! WGSL has no portable `f64`, so the GPU path is `f32`. Coordinates are
//! recentred on the scene bounding box before the cast, and sums are blocked,
//! which keeps the error against the `f64` CPU result at the 1e-6 level for
//! well-conditioned scenes (see `validate`).

use crate::{cpu, Segment};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shader.wgsl");
/// Points per dispatch; keeps buffers well under default binding limits.
const CHUNK: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Naive,
    Tiled,
}

impl Kernel {
    fn entry(self) -> &'static str {
        match self {
            Kernel::Naive => "naive",
            Kernel::Tiled => "tiled",
        }
    }
    pub fn label(self) -> &'static str {
        self.entry()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuConfig {
    pub kernel: Kernel,
    /// Threads per workgroup (power of two, <= device limit; 256 is always safe).
    pub workgroup_size: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            kernel: Kernel::Tiled,
            workgroup_size: 128,
        }
    }
}

/// Wall-clock breakdown of one GPU call, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTiming {
    pub upload_ms: f64,
    pub kernel_ms: f64,
    pub readback_ms: f64,
}

impl GpuTiming {
    pub fn total_ms(&self) -> f64 {
        self.upload_ms + self.kernel_ms + self.readback_ms
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSeg {
    a: [f32; 4],
    b: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_pts: u32,
    n_segs: u32,
    stride: u32,
    _pad: u32,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<(Kernel, u32), Arc<wgpu::ComputePipeline>>>,
    pub adapter_name: String,
    pub backend: String,
    pub max_workgroup_size: u32,
}

static GLOBAL: OnceLock<Result<GpuContext, String>> = OnceLock::new();

impl GpuContext {
    /// Lazily-created process-wide context.
    pub fn global() -> Result<&'static GpuContext, String> {
        GLOBAL
            .get_or_init(GpuContext::new)
            .as_ref()
            .map_err(|e| e.clone())
    }

    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or("no compatible GPU adapter found")?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fieldkernels"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |ro| wgpu::BufferBindingType::Storage { read_only: ro };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, wgpu::BufferBindingType::Uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let max_workgroup_size = device
            .limits()
            .max_compute_invocations_per_workgroup
            .min(device.limits().max_compute_workgroup_size_x);
        Ok(Self {
            device,
            queue,
            layout,
            pipeline_layout,
            pipelines: Mutex::new(HashMap::new()),
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            max_workgroup_size,
        })
    }

    fn pipeline(&self, kernel: Kernel, wg: u32) -> Arc<wgpu::ComputePipeline> {
        let mut map = self.pipelines.lock().unwrap();
        map.entry((kernel, wg))
            .or_insert_with(|| {
                let src = SHADER.replace("__WG__", &wg.to_string());
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("biot-savart"),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                Arc::new(
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(kernel.entry()),
                            layout: Some(&self.pipeline_layout),
                            module: &module,
                            entry_point: Some(kernel.entry()),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                )
            })
            .clone()
    }

    pub fn field(
        &self,
        points: &[[f64; 3]],
        segs: &[Segment],
        cfg: GpuConfig,
    ) -> Result<(Vec<[f64; 3]>, GpuTiming), String> {
        let wg = cfg.workgroup_size;
        if wg == 0 || !wg.is_power_of_two() || wg > self.max_workgroup_size {
            return Err(format!(
                "workgroup_size must be a power of two <= {}",
                self.max_workgroup_size
            ));
        }
        if points.is_empty() || segs.is_empty() {
            return Ok((vec![[0.0; 3]; points.len()], GpuTiming::default()));
        }
        let pipeline = self.pipeline(cfg.kernel, wg);
        let origin = cpu::centre(points, segs);
        let mut timing = GpuTiming::default();

        let t0 = Instant::now();
        let gpu_segs: Vec<GpuSeg> = segs
            .iter()
            .map(|s| GpuSeg {
                a: [
                    (s.a[0] - origin[0]) as f32,
                    (s.a[1] - origin[1]) as f32,
                    (s.a[2] - origin[2]) as f32,
                    s.current as f32,
                ],
                b: [
                    (s.b[0] - origin[0]) as f32,
                    (s.b[1] - origin[1]) as f32,
                    (s.b[2] - origin[2]) as f32,
                    0.0,
                ],
            })
            .collect();
        let seg_bytes = std::mem::size_of_val(gpu_segs.as_slice()) as u64;
        if seg_bytes > self.device.limits().max_storage_buffer_binding_size as u64 {
            return Err(format!(
                "{} filaments exceed the GPU storage-buffer limit",
                segs.len()
            ));
        }
        let seg_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("segs"),
                contents: bytemuck::cast_slice(&gpu_segs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        timing.upload_ms += ms(t0);

        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(CHUNK) {
            let t_up = Instant::now();
            let pts: Vec<[f32; 4]> = chunk
                .iter()
                .map(|p| {
                    [
                        (p[0] - origin[0]) as f32,
                        (p[1] - origin[1]) as f32,
                        (p[2] - origin[2]) as f32,
                        0.0,
                    ]
                })
                .collect();
            let bytes = (pts.len() * 16) as u64;
            let pts_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pts"),
                    contents: bytemuck::cast_slice(&pts),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let groups = (chunk.len() as u32).div_ceil(wg);
            let gx = groups.min(65535);
            let gy = groups.div_ceil(gx);
            let params = Params {
                n_pts: chunk.len() as u32,
                n_segs: segs.len() as u32,
                stride: gx * wg,
                _pad: 0,
            };
            let params_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: seg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            timing.upload_ms += ms(t_up);

            // Kernel: submit and wait on its own so its time is separable.
            let t_k = Instant::now();
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            self.queue.submit([enc.finish()]);
            self.device.poll(wgpu::Maintain::Wait);
            timing.kernel_ms += ms(t_k);

            let t_r = Instant::now();
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes);
            self.queue.submit([enc.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            read_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|e| e.to_string())?
                .map_err(|e| format!("buffer map failed: {e}"))?;
            {
                let view = read_buf.slice(..).get_mapped_range();
                let vals: &[[f32; 4]] = bytemuck::cast_slice(&view);
                out.extend(vals.iter().map(|v| [v[0] as f64, v[1] as f64, v[2] as f64]));
            }
            read_buf.unmap();
            timing.readback_ms += ms(t_r);
        }
        Ok((out, timing))
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
