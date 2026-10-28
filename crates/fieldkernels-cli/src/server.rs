//! Tiny local HTTP server: static frontend + JSON API. Binds to 127.0.0.1 only.

use fieldkernels_core::{bench, compute, geometry, validate, Backend, Segment};
use serde_json::{json, Value};
use std::io::Read;
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};

const INDEX: &str = include_str!("../../../frontend/index.html");
const STYLE: &str = include_str!("../../../frontend/style.css");
const APP: &str = include_str!("../../../frontend/app.js");

/// Heavy endpoints run one at a time so benchmarks are not skewed by overlap.
static HEAVY: Mutex<()> = Mutex::new(());

pub fn serve(port: u16, open: bool) -> ExitCode {
    let addr = format!("127.0.0.1:{port}");
    let server = match Server::http(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot bind {addr}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let url = format!("http://{addr}");
    println!("FieldKernels running at {url}  (Ctrl+C to stop)");
    println!(
        "GPU: {}",
        match fieldkernels_core::gpu::GpuContext::global() {
            Ok(g) => format!("{} via {}", g.adapter_name, g.backend),
            Err(e) => format!("unavailable ({e}); CPU backends only"),
        }
    );
    if open {
        let _ = if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/C", "start", "", &url])
                .spawn()
        } else if cfg!(target_os = "macos") {
            std::process::Command::new("open").arg(&url).spawn()
        } else {
            std::process::Command::new("xdg-open").arg(&url).spawn()
        };
    }
    for req in server.incoming_requests() {
        std::thread::spawn(move || handle(req));
    }
    ExitCode::SUCCESS
}

fn header(k: &str, v: &str) -> Header {
    Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn send(req: Request, status: u16, ctype: &str, body: String) {
    let _ = req.respond(
        Response::from_string(body)
            .with_status_code(status)
            .with_header(header("Content-Type", ctype))
            .with_header(header("Cache-Control", "no-store")),
    );
}

fn handle(mut req: Request) {
    let path = req.url().split('?').next().unwrap_or("/").to_string();
    let method = req.method().clone();
    if method == Method::Get {
        return match path.as_str() {
            "/" | "/index.html" => send(req, 200, "text/html; charset=utf-8", INDEX.into()),
            "/style.css" => send(req, 200, "text/css; charset=utf-8", STYLE.into()),
            "/app.js" => send(req, 200, "text/javascript; charset=utf-8", APP.into()),
            "/api/info" => send(req, 200, "application/json", info().to_string()),
            _ => send(req, 404, "text/plain", "not found".into()),
        };
    }
    if method == Method::Post {
        let mut body = String::new();
        if req
            .as_reader()
            .take(1 << 20)
            .read_to_string(&mut body)
            .is_err()
        {
            return send(
                req,
                400,
                "application/json",
                json!({"error": "bad body"}).to_string(),
            );
        }
        let input: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let _guard = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
        let result = match path.as_str() {
            "/api/field" => field(&input),
            "/api/bench" => run_bench(&input),
            "/api/validate" => Ok(run_validate()),
            _ => Err("unknown endpoint".to_string()),
        };
        return match result {
            Ok(v) => send(req, 200, "application/json", v.to_string()),
            Err(e) => send(
                req,
                400,
                "application/json",
                json!({ "error": e }).to_string(),
            ),
        };
    }
    send(req, 405, "text/plain", "method not allowed".into())
}

fn info() -> Value {
    let gpu = fieldkernels_core::gpu::GpuContext::global();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "cpu_threads": crate::rayon_threads(),
        "gpu": gpu.as_ref().ok().map(|g| json!({
            "name": g.adapter_name, "backend": g.backend, "max_workgroup_size": g.max_workgroup_size
        })),
        "gpu_error": gpu.as_ref().err(),
    })
}

fn num(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

/// Round to 4 significant digits so the JSON stays small and tidy.
fn sig4(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return 0.0;
    }
    format!("{x:.3e}").parse().unwrap_or(0.0)
}

fn field(input: &Value) -> Result<Value, String> {
    let preset = input
        .get("preset")
        .and_then(Value::as_str)
        .unwrap_or("helmholtz");
    let radius = num(input, "radius", 0.1).clamp(1e-3, 100.0);
    let current = num(input, "current", 10.0);
    let turns = num(input, "turns", 1.0).clamp(1.0, 500.0) as usize;
    let length = num(input, "length", 0.4).clamp(1e-3, 100.0);
    let per_turn = num(input, "per_turn", 128.0).clamp(8.0, 4096.0) as usize;
    let plane = input.get("plane").and_then(Value::as_str).unwrap_or("xz");
    let offset = num(input, "offset", 0.0);
    let extent = num(input, "extent", 0.25).clamp(1e-3, 1000.0);
    let res = num(input, "res", 128.0).clamp(8.0, 512.0) as usize;
    let backend_name = input
        .get("backend")
        .and_then(Value::as_str)
        .unwrap_or("cpu64");
    let mut backend = Backend::parse(backend_name).ok_or("unknown backend")?;

    let segs: Vec<Segment> = geometry::preset(preset, radius, current, turns, length, per_turn)
        .ok_or("unknown preset")?;
    if segs.len() > 2_000_000 {
        return Err("coil too large".into());
    }
    let pts = geometry::plane_grid(plane, offset, extent, res, res);

    let mut fallback = false;
    if matches!(backend, Backend::GpuNaive | Backend::GpuTiled)
        && fieldkernels_core::gpu::GpuContext::global().is_err()
    {
        backend = Backend::Cpu64;
        fallback = true;
    }
    let t = Instant::now();
    let b = compute(backend, &pts, &segs)?;
    let time_ms = t.elapsed().as_secs_f64() * 1e3;

    // in-plane components (u, v) and the axis normal to the slice
    let (iu, iv, iw) = match plane {
        "xy" => (0, 1, 2),
        "yz" => (1, 2, 0),
        _ => (0, 2, 1),
    };
    let ut = |v: f64| sig4(v * 1e6);
    let bmag: Vec<f64> = b
        .iter()
        .map(|v| ut((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()))
        .collect();
    let bu: Vec<f64> = b.iter().map(|v| ut(v[iu])).collect();
    let bv: Vec<f64> = b.iter().map(|v| ut(v[iv])).collect();
    let bw: Vec<f64> = b.iter().map(|v| ut(v[iw])).collect();

    let stride = (segs.len() / 6000).max(1);
    let coil: Vec<Value> = segs
        .iter()
        .step_by(stride)
        .map(|s| {
            json!([
                sig4(s.a[iu]),
                sig4(s.a[iv]),
                sig4(s.a[iw]),
                sig4(s.b[iu]),
                sig4(s.b[iv]),
                sig4(s.b[iw]),
                s.current.signum()
            ])
        })
        .collect();

    Ok(json!({
        "res": res, "extent": extent, "plane": plane, "offset": offset,
        "unit": "uT",
        "bmag": bmag, "bu": bu, "bv": bv, "bw": bw,
        "coil": coil,
        "n_filaments": segs.len(), "n_points": pts.len(),
        "backend": backend.label(), "fallback": fallback, "time_ms": time_ms,
    }))
}

fn run_bench(input: &Value) -> Result<Value, String> {
    let n_points = num(input, "points", 20_000.0).clamp(100.0, 2_000_000.0) as usize;
    let repeats = num(input, "repeats", 3.0).clamp(1.0, 10.0) as usize;
    let max_work = 4e10;
    let sizes: Vec<usize> = input
        .get("sizes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect()
        })
        .unwrap_or_else(|| vec![10, 100, 1_000, 10_000, 100_000]);
    let sizes: Vec<usize> = sizes
        .into_iter()
        .filter(|&n| n >= 1 && (n as f64) * (n_points as f64) <= max_work)
        .collect();
    if sizes.is_empty() {
        return Err("no sizes within the work limit (points x filaments <= 4e10)".into());
    }
    let rows = bench::run_sweep(&sizes, n_points, repeats);
    let cross = bench::crossover(&rows);
    let tune_n = sizes
        .iter()
        .copied()
        .filter(|&n| n <= 30_000)
        .max()
        .unwrap_or(1000)
        .max(1000);
    let tune = bench::workgroup_sweep(tune_n, n_points.min(50_000), repeats);
    Ok(json!({
        "points": n_points,
        "rows": rows.iter().map(|r| json!({
            "n": r.n_filaments, "cpu1": r.cpu_serial_ms, "cpu64": r.cpu64_ms, "cpu32": r.cpu32_ms,
            "gpu_naive": r.gpu_naive_ms, "gpu_tiled": r.gpu_tiled_ms, "gpu_kernel": r.gpu_tiled_kernel_ms,
            "speedup": r.speedup(), "kernel_speedup": r.kernel_speedup(),
        })).collect::<Vec<_>>(),
        "crossover": cross,
        "tune_filaments": tune_n,
        "tune": tune.iter().map(|t| json!({"kernel": t.kernel, "wg": t.workgroup_size, "ms": t.kernel_ms})).collect::<Vec<_>>(),
    }))
}

fn run_validate() -> Value {
    let checks = validate::run_all();
    json!({
        "checks": checks.iter().map(|c| json!({
            "name": c.name, "cases": c.cases, "max_rel_err": if c.max_rel_err.is_finite() { json!(c.max_rel_err) } else { Value::Null },
            "tolerance": c.tolerance, "passed": c.passed, "note": c.note,
        })).collect::<Vec<_>>(),
        "all_passed": checks.iter().all(|c| c.passed),
    })
}
