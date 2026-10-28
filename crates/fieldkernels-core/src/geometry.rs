//! Coil builders. Every coil is returned as a flat list of straight [`Segment`]s.

use crate::Segment;
use std::f64::consts::TAU;

/// Small deterministic xorshift64* generator (keeps the crate dependency-free
/// and makes benchmarks and tests reproducible).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Orthonormal basis (u, v) perpendicular to `normal`, right-handed with it.
pub fn basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = norm(normal);
    let helper = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = norm(cross(helper, n));
    let v = cross(n, u);
    (u, v)
}

/// Closed circular loop approximated by `n` inscribed chords. Current flows
/// counter-clockwise when viewed from the tip of `normal`.
pub fn circular_loop(
    center: [f64; 3],
    normal: [f64; 3],
    radius: f64,
    n: usize,
    current: f64,
) -> Vec<Segment> {
    let (u, v) = basis(normal);
    let pt = |k: usize| {
        let t = TAU * (k % n) as f64 / n as f64;
        let (s, c) = t.sin_cos();
        [
            center[0] + radius * (c * u[0] + s * v[0]),
            center[1] + radius * (c * u[1] + s * v[1]),
            center[2] + radius * (c * u[2] + s * v[2]),
        ]
    };
    (0..n)
        .map(|k| Segment::new(pt(k), pt(k + 1), current))
        .collect()
}

/// Two coaxial loops of radius `r` separated by `r` (uniform centre field).
/// `opposed = true` gives the anti-Helmholtz (gradient / quadrupole) pair.
pub fn helmholtz(r: f64, current: f64, turns: usize, n: usize, opposed: bool) -> Vec<Segment> {
    let mut out = Vec::new();
    for t in 0..turns.max(1) {
        // spread turns very slightly in radius so segments never coincide
        let rr = r * (1.0 + 1e-3 * t as f64);
        out.extend(circular_loop(
            [0.0, 0.0, -r / 2.0],
            [0.0, 0.0, 1.0],
            rr,
            n,
            current,
        ));
        let i2 = if opposed { -current } else { current };
        out.extend(circular_loop(
            [0.0, 0.0, r / 2.0],
            [0.0, 0.0, 1.0],
            rr,
            n,
            i2,
        ));
    }
    out
}

/// Helical solenoid along z, centred at the origin.
pub fn solenoid(
    radius: f64,
    length: f64,
    turns: usize,
    per_turn: usize,
    current: f64,
) -> Vec<Segment> {
    let total = turns * per_turn;
    let pt = |k: usize| {
        let f = k as f64 / total as f64;
        let t = TAU * turns as f64 * f;
        [radius * t.cos(), radius * t.sin(), length * (f - 0.5)]
    };
    (0..total)
        .map(|k| Segment::new(pt(k), pt(k + 1), current))
        .collect()
}

/// Rectangular (racetrack) loop in the xy-plane, `w` × `h`, corners rounded
/// with radius `corner`, discretised with `n` chords per corner.
pub fn racetrack(w: f64, h: f64, corner: f64, n: usize, current: f64) -> Vec<Segment> {
    let (hx, hy) = (w / 2.0 - corner, h / 2.0 - corner);
    let mut pts: Vec<[f64; 3]> = Vec::new();
    for (cx, cy, a0) in [
        (hx, hy, 0.0),
        (-hx, hy, 0.25),
        (-hx, -hy, 0.5),
        (hx, -hy, 0.75),
    ] {
        for k in 0..=n {
            let t = TAU * (a0 + 0.25 * k as f64 / n as f64);
            pts.push([cx + corner * t.cos(), cy + corner * t.sin(), 0.0]);
        }
    }
    (0..pts.len())
        .map(|k| Segment::new(pts[k], pts[(k + 1) % pts.len()], current))
        .collect()
}

/// Figure-eight (two counter-wound loops side by side along x).
pub fn figure_eight(r: f64, current: f64, n: usize) -> Vec<Segment> {
    let mut out = circular_loop([-r, 0.0, 0.0], [0.0, 0.0, 1.0], r, n, current);
    out.extend(circular_loop(
        [r, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        r,
        n,
        -current,
    ));
    out
}

/// A random-walk cloud of `n` chained short segments inside the unit cube with
/// alternating-sign currents. Not physical; a reproducible benchmark workload.
pub fn random_segments(n: usize, seed: u64) -> Vec<Segment> {
    let mut rng = Rng::new(seed);
    let mut cur = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let step = [
            rng.range(-0.05, 0.05),
            rng.range(-0.05, 0.05),
            rng.range(-0.05, 0.05),
        ];
        let next = [
            (cur[0] + step[0]).clamp(-1.0, 1.0),
            (cur[1] + step[1]).clamp(-1.0, 1.0),
            (cur[2] + step[2]).clamp(-1.0, 1.0),
        ];
        out.push(Segment::new(cur, next, rng.range(-1.0, 1.0)));
        cur = next;
    }
    out
}

/// Uniform random points in the cube `[-extent, extent]^3`.
pub fn random_points(n: usize, extent: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            [
                rng.range(-extent, extent),
                rng.range(-extent, extent),
                rng.range(-extent, extent),
            ]
        })
        .collect()
}

/// A regular grid of points on an axis-aligned plane. `plane` is "xy", "xz" or
/// "yz"; `offset` is the coordinate along the missing axis; `extent` is the
/// half-width. Returns points in row-major order (v rows, u columns) with `v`
/// increasing from `-extent` to `+extent`.
pub fn plane_grid(plane: &str, offset: f64, extent: f64, nu: usize, nv: usize) -> Vec<[f64; 3]> {
    let mut pts = Vec::with_capacity(nu * nv);
    for j in 0..nv {
        let v = -extent + 2.0 * extent * (j as f64 + 0.5) / nv as f64;
        for i in 0..nu {
            let u = -extent + 2.0 * extent * (i as f64 + 0.5) / nu as f64;
            pts.push(match plane {
                "xy" => [u, v, offset],
                "yz" => [offset, u, v],
                _ => [u, offset, v], // "xz"
            });
        }
    }
    pts
}

/// Named presets used by the web UI and the Python helpers.
pub fn preset(
    name: &str,
    radius: f64,
    current: f64,
    turns: usize,
    length: f64,
    per_turn: usize,
) -> Option<Vec<Segment>> {
    let turns = turns.max(1);
    let per_turn = per_turn.max(8);
    Some(match name {
        "loop" => helmholtz_free_loop(radius, current, turns, per_turn),
        "helmholtz" => helmholtz(radius, current, turns, per_turn, false),
        "anti-helmholtz" => helmholtz(radius, current, turns, per_turn, true),
        "solenoid" => solenoid(radius, length, turns, per_turn, current),
        "racetrack" => racetrack(
            2.0 * radius,
            radius,
            radius * 0.25,
            per_turn / 4 + 2,
            current,
        ),
        "figure-eight" => figure_eight(radius, current, per_turn),
        _ => return None,
    })
}

fn helmholtz_free_loop(r: f64, current: f64, turns: usize, n: usize) -> Vec<Segment> {
    (0..turns)
        .flat_map(|t| {
            circular_loop(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                r * (1.0 + 1e-3 * t as f64),
                n,
                current,
            )
        })
        .collect()
}
