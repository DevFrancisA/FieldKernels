//! PyO3 bindings. Exposed as `fieldkernels._native`; the friendly API lives in
//! `python/fieldkernels/__init__.py`.

use fieldkernels_core::{cpu, gpu, Segment};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn rows3(a: &PyReadonlyArray2<f64>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let v = a.as_array();
    if v.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (N, 3)"
        )));
    }
    Ok(v.outer_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn segments(
    starts: &PyReadonlyArray2<f64>,
    ends: &PyReadonlyArray2<f64>,
    currents: &PyReadonlyArray1<f64>,
) -> PyResult<Vec<Segment>> {
    let a = rows3(starts, "starts")?;
    let b = rows3(ends, "ends")?;
    let i = currents.as_array();
    if a.len() != b.len() || a.len() != i.len() {
        return Err(PyValueError::new_err(
            "starts, ends and currents must have the same length",
        ));
    }
    Ok(a.iter()
        .zip(&b)
        .zip(i.iter())
        .map(|((a, b), &i)| Segment::new(*a, *b, i))
        .collect())
}

/// CPU Biot-Savart field (tesla), shape (N, 3). `precision` is "f64" or "f32".
#[pyfunction]
#[pyo3(signature = (points, starts, ends, currents, precision="f64", parallel=true))]
fn biot_savart_cpu<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    starts: PyReadonlyArray2<'py, f64>,
    ends: PyReadonlyArray2<'py, f64>,
    currents: PyReadonlyArray1<'py, f64>,
    precision: &str,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = rows3(&points, "points")?;
    let segs = segments(&starts, &ends, &currents)?;
    let out = py.allow_threads(|| match (precision, parallel) {
        ("f64", true) => Ok(cpu::field_f64(&pts, &segs)),
        ("f64", false) => Ok(cpu::field_f64_serial(&pts, &segs)),
        ("f32", true) => Ok(cpu::field_f32(&pts, &segs)),
        ("f32", false) => Ok(cpu::k32::field_serial(&pts, &segs)),
        _ => Err("precision must be 'f64' or 'f32'"),
    });
    let out = out.map_err(PyValueError::new_err)?;
    Ok(
        PyArray2::from_vec2(py, &out.iter().map(|v| v.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
}

/// GPU (wgpu, f32) Biot-Savart field in tesla. Returns (field, timing_ms dict).
#[pyfunction]
#[pyo3(signature = (points, starts, ends, currents, kernel="tiled", workgroup_size=128))]
fn biot_savart_gpu<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    starts: PyReadonlyArray2<'py, f64>,
    ends: PyReadonlyArray2<'py, f64>,
    currents: PyReadonlyArray1<'py, f64>,
    kernel: &str,
    workgroup_size: u32,
) -> PyResult<(Bound<'py, PyArray2<f64>>, (f64, f64, f64))> {
    let pts = rows3(&points, "points")?;
    let segs = segments(&starts, &ends, &currents)?;
    let kernel = match kernel {
        "naive" => gpu::Kernel::Naive,
        "tiled" => gpu::Kernel::Tiled,
        _ => return Err(PyValueError::new_err("kernel must be 'naive' or 'tiled'")),
    };
    let ctx = gpu::GpuContext::global().map_err(PyRuntimeError::new_err)?;
    let (out, t) = py
        .allow_threads(|| {
            ctx.field(
                &pts,
                &segs,
                gpu::GpuConfig {
                    kernel,
                    workgroup_size,
                },
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let arr = PyArray2::from_vec2(py, &out.iter().map(|v| v.to_vec()).collect::<Vec<_>>())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((arr, (t.upload_ms, t.kernel_ms, t.readback_ms)))
}

/// Name of the GPU adapter, or None if no adapter is usable.
#[pyfunction]
fn gpu_info() -> Option<String> {
    gpu::GpuContext::global()
        .ok()
        .map(|g| format!("{} ({})", g.adapter_name, g.backend))
}

/// Circular loop as (starts, ends, currents) arrays.
#[pyfunction]
#[pyo3(signature = (center, normal, radius, n, current))]
fn loop_filaments<'py>(
    py: Python<'py>,
    center: [f64; 3],
    normal: [f64; 3],
    radius: f64,
    n: usize,
    current: f64,
) -> (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Vec<f64>,
) {
    let segs = fieldkernels_core::geometry::circular_loop(center, normal, radius, n, current);
    let a: Vec<[f64; 3]> = segs.iter().map(|s| s.a).collect();
    let b: Vec<[f64; 3]> = segs.iter().map(|s| s.b).collect();
    (
        PyArray2::from_vec2(py, &a.iter().map(|v| v.to_vec()).collect::<Vec<_>>()).unwrap(),
        PyArray2::from_vec2(py, &b.iter().map(|v| v.to_vec()).collect::<Vec<_>>()).unwrap(),
        vec![current; n],
    )
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(biot_savart_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(biot_savart_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(loop_filaments, m)?)?;
    Ok(())
}
