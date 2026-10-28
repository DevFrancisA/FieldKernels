//! CPU Biot-Savart kernels.
//!
//! For a straight segment `a -> b` carrying current `I`, with `dl = b - a`,
//! `r1 = p - a`, `r2 = p - b` and `c = dl x r1`, the exact field is
//!
//! ```text
//! B = (mu0 I / 4 pi) * c / |c|^2 * ( dl.r1/|r1| - dl.r2/|r2| )
//! ```
//!
//! Points within `sqrt(EPS)·|dl|` of the wire's line contribute zero for that
//! segment (the field diverges on the wire itself).

use crate::{Segment, MU0_OVER_4PI};
use rayon::prelude::*;

macro_rules! kernel_mod {
    ($m:ident, $t:ty, $eps:expr) => {
        pub mod $m {
            use super::*;
            pub type R = $t;
            const EPS: R = $eps;

            #[derive(Clone, Copy)]
            pub struct Seg {
                pub a: [R; 3],
                pub b: [R; 3],
                pub i: R,
            }

            /// Convert (and optionally recentre) segments into this precision.
            pub fn convert(segs: &[Segment], origin: [f64; 3]) -> Vec<Seg> {
                segs.iter()
                    .map(|s| Seg {
                        a: [
                            (s.a[0] - origin[0]) as R,
                            (s.a[1] - origin[1]) as R,
                            (s.a[2] - origin[2]) as R,
                        ],
                        b: [
                            (s.b[0] - origin[0]) as R,
                            (s.b[1] - origin[1]) as R,
                            (s.b[2] - origin[2]) as R,
                        ],
                        i: s.current as R,
                    })
                    .collect()
            }

            /// Field (tesla) at one point, summed over all segments.
            #[inline]
            pub fn point_field(p: [R; 3], segs: &[Seg]) -> [R; 3] {
                let mut acc = [0.0 as R; 3];
                // Sum in blocks so rounding error grows ~ sqrt(blocks) not N.
                for block in segs.chunks(64) {
                    let mut part = [0.0 as R; 3];
                    for s in block {
                        let dl = [s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2]];
                        let r1 = [p[0] - s.a[0], p[1] - s.a[1], p[2] - s.a[2]];
                        let r2 = [p[0] - s.b[0], p[1] - s.b[1], p[2] - s.b[2]];
                        let l2 = dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2];
                        let c = [
                            dl[1] * r1[2] - dl[2] * r1[1],
                            dl[2] * r1[0] - dl[0] * r1[2],
                            dl[0] * r1[1] - dl[1] * r1[0],
                        ];
                        let c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                        if l2 <= 0.0 || c2 <= EPS * l2 * l2 {
                            continue;
                        }
                        let n1 = (r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]).sqrt();
                        let n2 = (r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]).sqrt();
                        let t = (dl[0] * r1[0] + dl[1] * r1[1] + dl[2] * r1[2]) / n1
                            - (dl[0] * r2[0] + dl[1] * r2[1] + dl[2] * r2[2]) / n2;
                        let k = s.i * t / c2;
                        part[0] += k * c[0];
                        part[1] += k * c[1];
                        part[2] += k * c[2];
                    }
                    acc[0] += part[0];
                    acc[1] += part[1];
                    acc[2] += part[2];
                }
                let m = MU0_OVER_4PI as R;
                [acc[0] * m, acc[1] * m, acc[2] * m]
            }

            fn run(points: &[[f64; 3]], segs: &[Segment], parallel: bool) -> Vec<[f64; 3]> {
                let origin = super::centre(points, segs);
                let cs = convert(segs, origin);
                let eval = |p: &[f64; 3]| {
                    let q = [
                        (p[0] - origin[0]) as R,
                        (p[1] - origin[1]) as R,
                        (p[2] - origin[2]) as R,
                    ];
                    let b = point_field(q, &cs);
                    [b[0] as f64, b[1] as f64, b[2] as f64]
                };
                if parallel {
                    points.par_iter().map(eval).collect()
                } else {
                    points.iter().map(eval).collect()
                }
            }

            /// Multithreaded (Rayon) evaluation over points.
            pub fn field(points: &[[f64; 3]], segs: &[Segment]) -> Vec<[f64; 3]> {
                run(points, segs, true)
            }
            /// Single-threaded evaluation (baseline for scaling studies).
            pub fn field_serial(points: &[[f64; 3]], segs: &[Segment]) -> Vec<[f64; 3]> {
                run(points, segs, false)
            }
        }
    };
}

/// Midpoint of the bounding box of everything; subtracting it before casting
/// to `f32` keeps coordinates small and improves single-precision accuracy.
pub(crate) fn centre(points: &[[f64; 3]], segs: &[Segment]) -> [f64; 3] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut upd = |p: &[f64; 3]| {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    };
    points.iter().for_each(&mut upd);
    segs.iter().for_each(|s| {
        upd(&s.a);
        upd(&s.b);
    });
    if lo[0].is_infinite() {
        return [0.0; 3];
    }
    [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ]
}

kernel_mod!(k64, f64, 1e-24);
kernel_mod!(k32, f32, 1e-8);

pub fn field_f64(points: &[[f64; 3]], segs: &[Segment]) -> Vec<[f64; 3]> {
    k64::field(points, segs)
}
pub fn field_f32(points: &[[f64; 3]], segs: &[Segment]) -> Vec<[f64; 3]> {
    k32::field(points, segs)
}
pub fn field_f64_serial(points: &[[f64; 3]], segs: &[Segment]) -> Vec<[f64; 3]> {
    k64::field_serial(points, segs)
}
