//! Correctness suite: every kernel is checked against an independent reference.
//!
//! References, from strongest to weakest:
//! 1. exact on-axis field of a regular polygon (isolates arithmetic error),
//! 2. elliptic-integral solution of a circular loop (includes discretisation),
//! 3. infinite straight wire limit,
//! 4. div B = 0 by central differences,
//! 5. GPU (f32) vs CPU (f64) on the same physical coils.

use crate::{analytic, cpu, geometry, Segment};

#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    pub cases: usize,
    /// Maximum relative error observed (dimensionless).
    pub max_rel_err: f64,
    pub tolerance: f64,
    pub passed: bool,
    pub note: String,
}

fn check(name: &str, cases: usize, err: f64, tol: f64, note: &str) -> Check {
    Check {
        name: name.into(),
        cases,
        max_rel_err: err,
        tolerance: tol,
        passed: err.is_finite() && err <= tol,
        note: note.into(),
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Max over points of |a-b| / max(|b|, floor) where floor = 1e-3 * max|b|.
/// The floor stops points where the true field nearly cancels from dominating.
pub fn max_scaled_err(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let bmax = b.iter().map(|v| norm(*v)).fold(0.0, f64::max);
    let floor = 1e-3 * bmax;
    a.iter()
        .zip(b)
        .map(|(x, y)| norm([x[0] - y[0], x[1] - y[1], x[2] - y[2]]) / norm(*y).max(floor))
        .fold(0.0, f64::max)
}

type Kernel = fn(&[[f64; 3]], &[Segment]) -> Vec<[f64; 3]>;

fn polygon_check(name: &str, kernel: Kernel, tol: f64) -> Check {
    let mut rng = geometry::Rng::new(11);
    let mut worst: f64 = 0.0;
    let cases = 200;
    for _ in 0..cases {
        let n = 8 + (rng.next_u64() % 505) as usize;
        let r = rng.range(0.05, 2.0);
        let i = rng.range(0.5, 50.0);
        let z = rng.range(-2.0, 2.0) * r;
        let segs = geometry::circular_loop([0.0; 3], [0.0, 0.0, 1.0], r, n, i);
        let b = kernel(&[[0.0, 0.0, z]], &segs)[0];
        let exact = analytic::polygon_on_axis(n, r, i, z);
        worst = worst.max((b[2] - exact).abs() / exact.abs());
        // Transverse components must vanish on axis relative to |Bz|.
        worst = worst.max(b[0].abs().max(b[1].abs()) / exact.abs());
    }
    check(
        name,
        cases,
        worst,
        tol,
        "exact regular-polygon on-axis field",
    )
}

fn loop_check() -> Check {
    let n = 8192;
    let (a, i) = (0.5, 3.0);
    let segs = geometry::circular_loop([0.0; 3], [0.0, 0.0, 1.0], a, n, i);
    let mut rng = geometry::Rng::new(5);
    let mut pts = Vec::new();
    let mut refs = Vec::new();
    while pts.len() < 500 {
        let rho = rng.range(0.0, 1.5 * a);
        let z = rng.range(-1.5 * a, 1.5 * a);
        // stay >= 0.1 a away from the wire, where chord discretisation dominates
        if ((rho - a).powi(2) + z * z).sqrt() < 0.1 * a {
            continue;
        }
        let phi = rng.range(0.0, std::f64::consts::TAU);
        let (br, bz) = analytic::loop_field_cyl(a, i, rho, z);
        pts.push([rho * phi.cos(), rho * phi.sin(), z]);
        refs.push([br * phi.cos(), br * phi.sin(), bz]);
    }
    let got = cpu::field_f64(&pts, &segs);
    check(
        "circular loop vs elliptic integrals (N=8192 chords)",
        pts.len(),
        max_scaled_err(&got, &refs),
        2e-6,
        "includes O((pi/N)^2) chord error",
    )
}

fn wire_check() -> Check {
    let len = 2.0e4;
    let seg = [Segment::new(
        [-len / 2.0, 0.0, 0.0],
        [len / 2.0, 0.0, 0.0],
        7.0,
    )];
    let mut worst: f64 = 0.0;
    let ds = [0.05, 0.3, 1.0, 2.5];
    for &d in &ds {
        let b = cpu::field_f64(&[[0.0, d, 0.0]], &seg)[0];
        // current +x, point +y => B along +z
        worst = worst.max((b[2] - analytic::infinite_wire(7.0, d)).abs() / b[2].abs());
        worst = worst.max(b[0].abs().max(b[1].abs()) / b[2].abs());
    }
    check(
        "long wire vs mu0 I / 2 pi d",
        ds.len(),
        worst,
        1e-6,
        "finite-length truncation ~ (d/L)^2",
    )
}

fn divergence_check() -> Check {
    let segs = geometry::helmholtz(0.4, 5.0, 1, 512, false);
    let h = 1e-4;
    let mut rng = geometry::Rng::new(3);
    let mut worst: f64 = 0.0;
    let cases = 100;
    for _ in 0..cases {
        let p = [
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
            rng.range(-0.15, 0.15),
        ];
        let mut div = 0.0;
        for k in 0..3 {
            let mut hi = p;
            let mut lo = p;
            hi[k] += h;
            lo[k] -= h;
            let f = cpu::field_f64(&[hi, lo], &segs);
            div += (f[0][k] - f[1][k]) / (2.0 * h);
        }
        let scale = norm(cpu::field_f64(&[p], &segs)[0]) / 0.4; // |B| / R
        worst = worst.max(div.abs() / scale);
    }
    check(
        "div B = 0 (central differences, f64)",
        cases,
        worst,
        1e-6,
        "|div B| / (|B|/R)",
    )
}

fn coils() -> Vec<(&'static str, Vec<Segment>)> {
    vec![
        (
            "loop",
            geometry::circular_loop([0.0; 3], [0.0, 0.0, 1.0], 0.5, 256, 4.0),
        ),
        ("helmholtz", geometry::helmholtz(0.5, 4.0, 3, 128, false)),
        (
            "anti-helmholtz",
            geometry::helmholtz(0.5, 4.0, 2, 128, true),
        ),
        ("solenoid", geometry::solenoid(0.2, 1.0, 30, 64, 2.0)),
        ("racetrack", geometry::racetrack(1.2, 0.6, 0.15, 24, 3.0)),
    ]
}

fn scene_points(seed: u64) -> Vec<[f64; 3]> {
    let mut pts = geometry::random_points(3000, 0.9, seed);
    pts.extend(geometry::plane_grid("xz", 0.0, 0.9, 40, 40));
    pts
}

fn f32_cpu_check() -> Check {
    let mut worst: f64 = 0.0;
    let mut cases = 0;
    for (_, segs) in coils() {
        let pts = scene_points(21);
        worst = worst.max(max_scaled_err(
            &cpu::field_f32(&pts, &segs),
            &cpu::field_f64(&pts, &segs),
        ));
        cases += pts.len();
    }
    check(
        "CPU f32 vs CPU f64 (5 coil types)",
        cases,
        worst,
        1e-4,
        "single-precision rounding",
    )
}

#[cfg(feature = "gpu")]
fn gpu_checks() -> Vec<Check> {
    use crate::gpu::{GpuConfig, GpuContext, Kernel};
    let ctx = match GpuContext::global() {
        Ok(c) => c,
        Err(e) => {
            return vec![Check {
                name: "GPU kernels".into(),
                cases: 0,
                max_rel_err: f64::NAN,
                tolerance: 0.0,
                passed: false,
                note: format!("skipped: {e}"),
            }]
        }
    };
    let mut out = Vec::new();
    for kernel in [Kernel::Naive, Kernel::Tiled] {
        let mut worst: f64 = 0.0;
        let mut cases = 0;
        for (_, segs) in coils() {
            let pts = scene_points(33);
            let cfg = GpuConfig {
                kernel,
                workgroup_size: 64,
            };
            match ctx.field(&pts, &segs, cfg) {
                Ok((g, _)) => worst = worst.max(max_scaled_err(&g, &cpu::field_f64(&pts, &segs))),
                Err(e) => {
                    out.push(check(
                        &format!("GPU {}", kernel.label()),
                        0,
                        f64::NAN,
                        0.0,
                        &e,
                    ));
                    return out;
                }
            }
            cases += pts.len();
        }
        out.push(check(
            &format!(
                "GPU {} (f32) vs CPU f64, {}",
                kernel.label(),
                ctx.adapter_name
            ),
            cases,
            worst,
            1e-4,
            "5 coil types, 3000 random + 1600 grid points each",
        ));
    }
    // Awkward sizes: point/segment counts that are not multiples of the workgroup size.
    let segs = geometry::circular_loop([0.0; 3], [0.0, 0.0, 1.0], 0.5, 1001, 1.0);
    let pts = geometry::random_points(777, 0.8, 4);
    let cfg = GpuConfig {
        kernel: Kernel::Tiled,
        workgroup_size: 256,
    };
    if let Ok((g, _)) = ctx.field(&pts, &segs, cfg) {
        out.push(check(
            "GPU tiled, ragged sizes (777 pts, 1001 segs, wg=256)",
            pts.len(),
            max_scaled_err(&g, &cpu::field_f64(&pts, &segs)),
            1e-4,
            "tile and dispatch edge handling",
        ));
    }
    out
}

/// Run the whole suite.
pub fn run_all() -> Vec<Check> {
    let mut v = vec![
        polygon_check("CPU f64 vs exact polygon on-axis", cpu::field_f64, 1e-12),
        polygon_check("CPU f32 vs exact polygon on-axis", cpu::field_f32, 1e-4),
        loop_check(),
        wire_check(),
        divergence_check(),
        f32_cpu_check(),
    ];
    #[cfg(feature = "gpu")]
    v.extend(gpu_checks());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_suite_passes() {
        for c in [
            polygon_check("p64", cpu::field_f64, 1e-12),
            polygon_check("p32", cpu::field_f32, 1e-4),
            loop_check(),
            wire_check(),
            divergence_check(),
            f32_cpu_check(),
        ] {
            assert!(
                c.passed,
                "{}: {:e} > {:e}",
                c.name, c.max_rel_err, c.tolerance
            );
        }
    }

    #[test]
    fn serial_equals_parallel() {
        let segs = geometry::random_segments(500, 1);
        let pts = geometry::random_points(300, 1.2, 2);
        assert_eq!(
            cpu::field_f64(&pts, &segs),
            cpu::field_f64_serial(&pts, &segs)
        );
    }

    #[test]
    fn point_on_wire_is_finite() {
        let segs = [Segment::new([0.0; 3], [1.0, 0.0, 0.0], 1.0)];
        let b = cpu::field_f64(&[[0.5, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0]], &segs);
        assert!(b.iter().flatten().all(|v| v.is_finite()));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu_when_available() {
        if crate::gpu::GpuContext::global().is_err() {
            eprintln!("no GPU adapter; skipping");
            return;
        }
        for c in gpu_checks() {
            assert!(
                c.passed,
                "{}: {:e} > {:e} ({})",
                c.name, c.max_rel_err, c.tolerance, c.note
            );
        }
    }
}
