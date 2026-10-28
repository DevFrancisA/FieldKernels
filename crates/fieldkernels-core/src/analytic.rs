//! Closed-form reference solutions used to validate the numerical kernels.

use crate::{MU0, MU0_OVER_4PI};
use std::f64::consts::PI;

/// Complete elliptic integrals K(m), E(m) with parameter `m = k^2`, via the
/// arithmetic-geometric mean (accurate to ~1e-16 for 0 <= m < 1).
pub fn ellip_ke(m: f64) -> (f64, f64) {
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    let mut c = m.sqrt();
    let mut sum = 0.5 * c * c; // 2^(n-1) c_n^2 at n = 0
    let mut pow = 0.5;
    for _ in 0..40 {
        let an = 0.5 * (a + b);
        c = 0.5 * (a - b);
        b = (a * b).sqrt();
        a = an;
        pow *= 2.0;
        sum += pow * c * c;
        if c.abs() < 1e-17 {
            break;
        }
    }
    let k = PI / (2.0 * a);
    (k, k * (1.0 - sum))
}

/// Field of a circular loop (radius `a`, current `i`, centred at the origin in
/// the xy-plane, current counter-clockwise seen from +z) at cylindrical
/// coordinates `(rho, z)`. Returns `(B_rho, B_z)` in tesla (Smythe / Jackson).
pub fn loop_field_cyl(a: f64, i: f64, rho: f64, z: f64) -> (f64, f64) {
    let alpha2 = a * a + rho * rho + z * z - 2.0 * a * rho;
    let beta2 = a * a + rho * rho + z * z + 2.0 * a * rho;
    let m = 1.0 - alpha2 / beta2;
    let (k, e) = ellip_ke(m);
    let beta = beta2.sqrt();
    let pref = MU0 * i / (2.0 * PI);
    let bz = pref / beta * (k + (a * a - rho * rho - z * z) / alpha2 * e);
    let brho = if rho.abs() < 1e-12 * a {
        0.0
    } else {
        pref * z / (rho * beta) * (-k + (a * a + rho * rho + z * z) / alpha2 * e)
    };
    (brho, bz)
}

/// On-axis field of a circular loop at height `z`.
pub fn loop_on_axis(a: f64, i: f64, z: f64) -> f64 {
    MU0 * i * a * a / (2.0 * (a * a + z * z).powf(1.5))
}

/// Exact on-axis B_z of a *regular N-gon* (circumradius `r`, inscribed in the
/// circle, current `i`) at height `z`. This is the exact answer for a polygonal
/// coil, so it isolates kernel arithmetic error from discretisation error.
pub fn polygon_on_axis(n: usize, r: f64, i: f64, z: f64) -> f64 {
    let n_f = n as f64;
    let half_len = r * (PI / n_f).sin();
    let apothem = r * (PI / n_f).cos();
    let rho = (apothem * apothem + z * z).sqrt();
    let b_side = MU0_OVER_4PI * i / rho * 2.0 * half_len / (rho * rho + half_len * half_len).sqrt();
    n_f * b_side * apothem / rho
}

/// Field magnitude of an infinitely long straight wire at distance `d`.
pub fn infinite_wire(i: f64, d: f64) -> f64 {
    MU0 * i / (2.0 * PI * d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptic_known_values() {
        let (k, e) = ellip_ke(0.5);
        assert!((k - 1.854_074_677_301_372).abs() < 1e-14);
        assert!((e - 1.350_643_881_047_675).abs() < 1e-14);
        let (k0, e0) = ellip_ke(0.0);
        assert!((k0 - PI / 2.0).abs() < 1e-15 && (e0 - PI / 2.0).abs() < 1e-15);
    }

    #[test]
    fn off_axis_formula_matches_on_axis() {
        let (_, bz) = loop_field_cyl(0.3, 2.0, 1e-9, 0.17);
        assert!((bz / loop_on_axis(0.3, 2.0, 0.17) - 1.0).abs() < 1e-9);
    }
}
