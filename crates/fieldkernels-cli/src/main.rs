//! `fieldkernels` command line.
//!
//!   fieldkernels serve    [--port 8080] [--open]     local web app
//!   fieldkernels bench    [--points N] [--sizes a,b,c] [--repeats R] [--csv FILE]
//!   fieldkernels validate                            correctness suite
//!   fieldkernels info                                detected CPU / GPU

mod server;

use fieldkernels_core::{bench, validate};
use std::process::ExitCode;

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn gpu_line() -> String {
    match fieldkernels_core::gpu::GpuContext::global() {
        Ok(g) => format!(
            "{} ({}, max workgroup {})",
            g.adapter_name, g.backend, g.max_workgroup_size
        ),
        Err(e) => format!("none ({e})"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("serve") | None => {
            let port: u16 = arg(&args, "--port")
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080);
            server::serve(port, args.iter().any(|a| a == "--open"))
        }
        Some("info") => {
            println!("CPU threads : {}", rayon_threads());
            println!("GPU         : {}", gpu_line());
            ExitCode::SUCCESS
        }
        Some("validate") => {
            let checks = validate::run_all();
            let mut ok = true;
            for c in &checks {
                println!(
                    "[{}] {:<58} cases={:<6} max_rel_err={:.3e} (tol {:.0e})  {}",
                    if c.passed { "PASS" } else { "FAIL" },
                    c.name,
                    c.cases,
                    c.max_rel_err,
                    c.tolerance,
                    c.note
                );
                ok &= c.passed;
            }
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("bench") => {
            let points: usize = arg(&args, "--points")
                .and_then(|p| p.parse().ok())
                .unwrap_or(20_000);
            let repeats: usize = arg(&args, "--repeats")
                .and_then(|p| p.parse().ok())
                .unwrap_or(3);
            let sizes: Vec<usize> = arg(&args, "--sizes")
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_else(|| vec![10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]);
            println!(
                "CPU threads: {}   GPU: {}   points: {}\n",
                rayon_threads(),
                gpu_line(),
                points
            );
            let rows = bench::run_sweep(&sizes, points, repeats);
            let fmt = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.2}"));
            println!(
                "{:>10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>9} {:>9}",
                "filaments",
                "cpu1_ms",
                "cpuN_f64_ms",
                "cpuN_f32_ms",
                "gpu_naive",
                "gpu_tiled",
                "gpu_kernel",
                "speedup",
                "kernel_x"
            );
            for r in &rows {
                println!(
                    "{:>10} {:>11} {:>11.2} {:>11.2} {:>11} {:>11} {:>11} {:>9} {:>9}",
                    r.n_filaments,
                    fmt(r.cpu_serial_ms),
                    r.cpu64_ms,
                    r.cpu32_ms,
                    fmt(r.gpu_naive_ms),
                    fmt(r.gpu_tiled_ms),
                    fmt(r.gpu_tiled_kernel_ms),
                    fmt(r.speedup()),
                    fmt(r.kernel_speedup()),
                );
            }
            match bench::crossover(&rows) {
                Some(n) => println!(
                    "\nGPU (end-to-end) overtakes the multithreaded CPU from {n} filaments."
                ),
                None => println!(
                    "\nGPU (end-to-end) did not beat the multithreaded CPU at any measured size."
                ),
            }
            if let Some(last) = rows.last() {
                println!(
                    "Peak CPU throughput {:.2} G interactions/s; peak GPU kernel {:.2} G/s.",
                    rows.iter()
                        .map(|r| bench::ginteractions_per_s(r.n_filaments, r.n_points, r.cpu64_ms))
                        .fold(0.0, f64::max),
                    rows.iter()
                        .filter_map(
                            |r| r.gpu_tiled_kernel_ms.map(|k| bench::ginteractions_per_s(
                                r.n_filaments,
                                last.n_points,
                                k
                            ))
                        )
                        .fold(0.0, f64::max)
                );
            }
            let tune_n = sizes
                .iter()
                .copied()
                .filter(|&n| n <= 30_000)
                .max()
                .unwrap_or(1000)
                .max(1000);
            let tune = bench::workgroup_sweep(tune_n, points, repeats);
            if !tune.is_empty() {
                println!("\nWorkgroup tuning ({} filaments, kernel-only ms):", tune_n);
                for t in &tune {
                    println!(
                        "  {:<6} wg={:<4} {:.3}",
                        t.kernel, t.workgroup_size, t.kernel_ms
                    );
                }
            }
            if let Some(path) = arg(&args, "--csv") {
                let mut s = String::from("filaments,points,cpu_serial_ms,cpu64_ms,cpu32_ms,gpu_naive_ms,gpu_tiled_ms,gpu_tiled_kernel_ms\n");
                let o = |v: Option<f64>| v.map_or(String::new(), |x| x.to_string());
                for r in &rows {
                    s += &format!(
                        "{},{},{},{},{},{},{},{}\n",
                        r.n_filaments,
                        r.n_points,
                        o(r.cpu_serial_ms),
                        r.cpu64_ms,
                        r.cpu32_ms,
                        o(r.gpu_naive_ms),
                        o(r.gpu_tiled_ms),
                        o(r.gpu_tiled_kernel_ms)
                    );
                }
                std::fs::write(path, s).expect("write csv");
                println!("\nwrote {path}");
            }
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command `{other}`. usage: fieldkernels [serve|bench|validate|info]");
            ExitCode::FAILURE
        }
    }
}

pub fn rayon_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
