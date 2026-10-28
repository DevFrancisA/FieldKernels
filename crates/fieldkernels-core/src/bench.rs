//! Benchmark harness: CPU vs GPU scaling, crossover size, workgroup tuning.
//!
//! Every number is measured wall-clock on the machine running it. "GPU total"
//! includes upload and readback; "GPU kernel" is dispatch-to-completion only.

use crate::{cpu, geometry};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct BenchRow {
    pub n_filaments: usize,
    pub n_points: usize,
    pub cpu_serial_ms: Option<f64>,
    pub cpu64_ms: f64,
    pub cpu32_ms: f64,
    pub gpu_naive_ms: Option<f64>,
    pub gpu_tiled_ms: Option<f64>,
    pub gpu_tiled_kernel_ms: Option<f64>,
}

impl BenchRow {
    /// End-to-end speedup of the tiled GPU kernel over multithreaded f64 CPU.
    pub fn speedup(&self) -> Option<f64> {
        self.gpu_tiled_ms.map(|g| self.cpu64_ms / g)
    }
    /// Same, counting the kernel alone (buffers already resident).
    pub fn kernel_speedup(&self) -> Option<f64> {
        self.gpu_tiled_kernel_ms.map(|g| self.cpu64_ms / g)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms<T>(repeats: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        std::hint::black_box(f());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

#[cfg(feature = "gpu")]
fn gpu_time(
    ctx: &crate::gpu::GpuContext,
    pts: &[[f64; 3]],
    segs: &[crate::Segment],
    cfg: crate::gpu::GpuConfig,
    repeats: usize,
) -> Option<(f64, f64)> {
    ctx.field(pts, segs, cfg).ok()?; // warm-up: pipeline compile, driver caches
    let mut total = Vec::new();
    let mut kern = Vec::new();
    for _ in 0..repeats {
        let (_, t) = ctx.field(pts, segs, cfg).ok()?;
        total.push(t.total_ms());
        kern.push(t.kernel_ms);
    }
    Some((median(total), median(kern)))
}

/// Sweep the number of filaments at a fixed number of field points.
pub fn run_sweep(sizes: &[usize], n_points: usize, repeats: usize) -> Vec<BenchRow> {
    let pts = geometry::random_points(n_points, 1.5, 99);
    #[cfg(feature = "gpu")]
    let ctx = crate::gpu::GpuContext::global().ok();
    sizes
        .iter()
        .map(|&n| {
            let segs = geometry::random_segments(n, 7);
            // Do less repeating for expensive cells.
            let work = (n as f64) * (n_points as f64);
            let reps = if work > 2e9 { 1 } else { repeats.max(1) };
            let mut row = BenchRow {
                n_filaments: n,
                n_points,
                ..Default::default()
            };
            if work <= 2e8 {
                row.cpu_serial_ms = Some(time_ms(reps, || cpu::field_f64_serial(&pts, &segs)));
            }
            row.cpu64_ms = time_ms(reps, || cpu::field_f64(&pts, &segs));
            row.cpu32_ms = time_ms(reps, || cpu::field_f32(&pts, &segs));
            #[cfg(feature = "gpu")]
            if let Some(ctx) = ctx {
                use crate::gpu::{GpuConfig, Kernel};
                let naive = GpuConfig {
                    kernel: Kernel::Naive,
                    workgroup_size: 128,
                };
                let tiled = GpuConfig {
                    kernel: Kernel::Tiled,
                    workgroup_size: 128,
                };
                row.gpu_naive_ms = gpu_time(ctx, &pts, &segs, naive, reps).map(|t| t.0);
                if let Some((t, k)) = gpu_time(ctx, &pts, &segs, tiled, reps) {
                    row.gpu_tiled_ms = Some(t);
                    row.gpu_tiled_kernel_ms = Some(k);
                }
            }
            row
        })
        .collect()
}

/// Smallest filament count from which the tiled GPU (end to end) beats the
/// multithreaded CPU for every larger size measured. `None` if it never does.
pub fn crossover(rows: &[BenchRow]) -> Option<usize> {
    let mut found = None;
    for r in rows.iter().rev() {
        match r.speedup() {
            Some(s) if s > 1.0 => found = Some(r.n_filaments),
            _ => break,
        }
    }
    found
}

#[derive(Clone, Debug)]
pub struct TuneRow {
    pub kernel: &'static str,
    pub workgroup_size: u32,
    pub kernel_ms: f64,
}

/// Time both kernels over several workgroup sizes (kernel-only time).
#[cfg(feature = "gpu")]
pub fn workgroup_sweep(n_filaments: usize, n_points: usize, repeats: usize) -> Vec<TuneRow> {
    use crate::gpu::{GpuConfig, GpuContext, Kernel};
    let Ok(ctx) = GpuContext::global() else {
        return vec![];
    };
    let pts = geometry::random_points(n_points, 1.5, 99);
    let segs = geometry::random_segments(n_filaments, 7);
    let mut out = Vec::new();
    for kernel in [Kernel::Naive, Kernel::Tiled] {
        for wg in [32u32, 64, 128, 256] {
            if wg > ctx.max_workgroup_size {
                continue;
            }
            let cfg = GpuConfig {
                kernel,
                workgroup_size: wg,
            };
            if let Some((_, k)) = gpu_time(ctx, &pts, &segs, cfg, repeats.max(3)) {
                out.push(TuneRow {
                    kernel: kernel.label(),
                    workgroup_size: wg,
                    kernel_ms: k,
                });
            }
        }
    }
    out
}

/// Billions of filament-point interactions per second.
pub fn ginteractions_per_s(n_filaments: usize, n_points: usize, ms: f64) -> f64 {
    (n_filaments as f64 * n_points as f64) / (ms * 1e-3) / 1e9
}
