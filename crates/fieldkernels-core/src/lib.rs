//! FieldKernels core: Biot-Savart fields of straight current filaments.
//!
//! The field of a coil is the sum of the exact analytic field of each straight
//! segment, so accuracy is limited by how well segments approximate the wire,
//! not by quadrature.
//!
//! * [`cpu`]   – Rayon-parallel kernels in `f64` and `f32` (plus serial variants)
//! * [`gpu`]   – wgpu/WGSL compute kernels in `f32` (naive and shared-memory tiled)
//! * [`analytic`] – closed-form references (loop on/off axis, polygon, wire)
//! * [`validate`] – correctness suite comparing kernels against the references
//! * [`bench`] – CPU vs GPU timing sweeps, crossover size, workgroup tuning
//! * [`geometry`] – coil builders (loops, Helmholtz, solenoid, random clouds)

pub mod analytic;
pub mod bench;
pub mod cpu;
pub mod geometry;
pub mod validate;

#[cfg(feature = "gpu")]
pub mod gpu;

/// mu_0 / (4 pi) in T·m/A (the classical value; relative uncertainty ~1e-10).
pub const MU0_OVER_4PI: f64 = 1e-7;
/// mu_0 in T·m/A.
pub const MU0: f64 = 4.0 * std::f64::consts::PI * MU0_OVER_4PI;

/// A straight current-carrying filament from `a` to `b` carrying `current`
/// amperes (positive = flowing from `a` to `b`). Units are metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub current: f64,
}

impl Segment {
    pub fn new(a: [f64; 3], b: [f64; 3], current: f64) -> Self {
        Self { a, b, current }
    }
}

/// Which numeric backend computed a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu64,
    Cpu32,
    GpuNaive,
    GpuTiled,
}

impl Backend {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "cpu64" | "cpu" | "cpu-f64" => Some(Backend::Cpu64),
            "cpu32" | "cpu-f32" => Some(Backend::Cpu32),
            "gpu-naive" => Some(Backend::GpuNaive),
            "gpu" | "gpu-tiled" => Some(Backend::GpuTiled),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu64 => "cpu64",
            Backend::Cpu32 => "cpu32",
            Backend::GpuNaive => "gpu-naive",
            Backend::GpuTiled => "gpu-tiled",
        }
    }
}

/// Compute B (tesla) at `points` with the chosen backend. GPU backends fall back
/// to an error string if no adapter is available.
pub fn compute(
    backend: Backend,
    points: &[[f64; 3]],
    segs: &[Segment],
) -> Result<Vec<[f64; 3]>, String> {
    match backend {
        Backend::Cpu64 => Ok(cpu::field_f64(points, segs)),
        Backend::Cpu32 => Ok(cpu::field_f32(points, segs)),
        #[cfg(feature = "gpu")]
        Backend::GpuNaive | Backend::GpuTiled => {
            let ctx = gpu::GpuContext::global()?;
            let kernel = if backend == Backend::GpuNaive {
                gpu::Kernel::Naive
            } else {
                gpu::Kernel::Tiled
            };
            ctx.field(
                points,
                segs,
                gpu::GpuConfig {
                    kernel,
                    workgroup_size: 128,
                },
            )
            .map(|(v, _)| v)
        }
        #[cfg(not(feature = "gpu"))]
        _ => Err("built without the `gpu` feature".into()),
    }
}
