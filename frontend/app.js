"use strict";
// FieldKernels frontend: no build step, no dependencies.

const $ = (id) => document.getElementById(id);
const PRESETS = [
  ["helmholtz", "Helmholtz"], ["loop", "Single loop"], ["anti-helmholtz", "Anti-Helmholtz"],
  ["solenoid", "Solenoid"], ["racetrack", "Racetrack"], ["figure-eight", "Figure-eight"],
];
const state = { preset: "helmholtz", plane: "xz", data: null, info: null, busy: false, queued: false };

// ---------- helpers ----------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const fmtT = (uT) => { const a = Math.abs(uT); return a >= 1000 ? (uT / 1000).toPrecision(3) + " mT" : a >= 0.1 ? uT.toPrecision(3) + " µT" : (uT * 1000).toPrecision(3) + " nT"; };
const fmtMs = (ms) => (ms == null ? "–" : ms >= 100 ? ms.toFixed(0) : ms >= 1 ? ms.toFixed(2) : ms.toFixed(3));
const sci = (x) => (x == null ? "n/a" : x === 0 ? "0" : x.toExponential(2));
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

// inferno-like colormap
const STOPS = [[0,0,4],[31,12,72],[85,15,109],[136,34,106],[186,54,85],[227,89,51],[249,140,10],[249,201,50],[252,255,164]];
function cmap(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------- controls ----------
const SLIDERS = {
  radius: (v) => (v * 100).toFixed(1) + " cm", current: (v) => v.toFixed(1) + " A", turns: (v) => v,
  length: (v) => (v * 100).toFixed(0) + " cm", per_turn: (v) => v, offset: (v) => (v * 100).toFixed(1) + " cm", extent: (v) => (v * 100).toFixed(0) + " cm",
};
function params() {
  const g = (id) => parseFloat($(id).value);
  return {
    preset: state.preset, radius: g("radius"), current: g("current"), turns: g("turns"), length: g("length"),
    per_turn: g("per_turn"), plane: state.plane, offset: g("offset"), extent: g("extent"), res: g("res"), backend: $("backend").value,
  };
}
function syncLabels() {
  for (const [id, f] of Object.entries(SLIDERS)) $("o-" + id).textContent = f(parseFloat($(id).value));
  $("row-length").hidden = state.preset !== "solenoid";
  // Offset range tracks the view so you can always step through the whole volume.
  const e = parseFloat($("extent").value);
  $("offset").min = -e; $("offset").max = e;
}

function buildPresets() {
  const box = $("presets");
  PRESETS.forEach(([id, label]) => {
    const b = document.createElement("button");
    b.textContent = label; b.dataset.id = id; b.setAttribute("aria-pressed", id === state.preset);
    b.onclick = () => { state.preset = id; [...box.children].forEach((c) => c.setAttribute("aria-pressed", c.dataset.id === id));
      if (id === "solenoid" && +$("turns").value < 8) $("turns").value = 12;
      if (id !== "solenoid" && +$("turns").value > 20) $("turns").value = 1;
      syncLabels(); refresh(); };
    box.appendChild(b);
  });
  $("plane").querySelectorAll("button").forEach((b) => (b.onclick = () => {
    state.plane = b.dataset.v; $("plane").querySelectorAll("button").forEach((c) => c.setAttribute("aria-pressed", c === b)); refresh();
  }));
}

// ---------- compute + render ----------
async function refresh() {
  if (state.busy) { state.queued = true; return; }
  state.busy = true;
  try {
    const t0 = performance.now();
    const d = await api("/api/field", params());
    const rtt = performance.now() - t0;
    state.data = d;
    const rate = (d.n_filaments * d.n_points) / (d.time_ms * 1e-3) / 1e9;
    $("status").innerHTML = `<b>${d.n_filaments.toLocaleString()}</b> filaments × <b>${d.n_points.toLocaleString()}</b> points on <b>${d.backend}</b> in <b>${fmtMs(d.time_ms)} ms</b> (${rate.toFixed(2)} G interactions/s, round trip ${rtt.toFixed(0)} ms)` +
      (d.fallback ? ` <span class="warn">· GPU unavailable, used CPU</span>` : "");
    draw();
  } catch (e) {
    $("status").innerHTML = `<span class="warn">${e.message}</span>`;
  } finally {
    state.busy = false;
    if (state.queued) { state.queued = false; refresh(); }
  }
}
const refreshSoon = debounce(refresh, 90);

const view = $("view"), ctx = view.getContext("2d");
let off = null; // offscreen heatmap
let geom = null; // current mapping

function draw() {
  const d = state.data; if (!d) return;
  const dpr = window.devicePixelRatio || 1;
  const W = view.clientWidth, H = view.clientHeight;
  view.width = W * dpr; view.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#05070b"; ctx.fillRect(0, 0, W, H);
  const side = Math.min(W, H) - 24, x0 = (W - side) / 2, y0 = (H - side) / 2;
  geom = { x0, y0, side };

  // heat map
  const n = d.res, log = $("log").checked;
  const sorted = Float64Array.from(d.bmag).sort();
  const vmax = Math.max(sorted[Math.floor(sorted.length * 0.99)], 1e-9);
  const vmin = log ? vmax * 1e-3 : 0;
  geom.vmin = vmin; geom.vmax = vmax;
  const img = new ImageData(n, n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = d.bmag[j * n + i];
    const t = log ? (Math.log(Math.max(v, vmin)) - Math.log(vmin)) / (Math.log(vmax) - Math.log(vmin)) : v / vmax;
    const [r, g, b] = cmap(t);
    const k = ((n - 1 - j) * n + i) * 4; // v up
    img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
  }
  off = off || document.createElement("canvas"); off.width = n; off.height = n;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(off, x0, y0, side, side);
  ctx.strokeStyle = "#232c3b"; ctx.strokeRect(x0 + .5, y0 + .5, side, side);

  if ($("lines").checked) drawLines(d, n, x0, y0, side);
  if ($("coil").checked) drawCoil(d, x0, y0, side);
  drawAxes(d, x0, y0, side);
  drawLegend(vmin, vmax);
}

const toPx = (u, v, d, x0, y0, s) => [x0 + ((u + d.extent) / (2 * d.extent)) * s, y0 + (1 - (v + d.extent) / (2 * d.extent)) * s];

function sample(arr, n, fi, fj) {
  fi = Math.min(n - 1.001, Math.max(0, fi)); fj = Math.min(n - 1.001, Math.max(0, fj));
  const i = Math.floor(fi), j = Math.floor(fj), a = fi - i, b = fj - j;
  const g = (jj, ii) => arr[jj * n + ii];
  return (g(j, i) * (1 - a) + g(j, i + 1) * a) * (1 - b) + (g(j + 1, i) * (1 - a) + g(j + 1, i + 1) * a) * b;
}

function drawLines(d, n, x0, y0, s) {
  ctx.save(); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,.4)";
  const vecAt = (gx, gy) => { // grid coords (cell units)
    const fi = gx - 0.5, fj = gy - 0.5;
    return [sample(d.bu, n, fi, fj), sample(d.bv, n, fi, fj)];
  };
  const k = 11, step = 0.7, maxSteps = 500;
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
    const sx = ((a + 0.5) / k) * n, sy = ((b + 0.5) / k) * n;
    for (const dir of [1, -1]) {
      let x = sx, y = sy; ctx.beginPath(); ctx.moveTo(x0 + (x / n) * s, y0 + (1 - y / n) * s);
      for (let m = 0; m < maxSteps; m++) {
        const [u1, v1] = vecAt(x, y), m1 = Math.hypot(u1, v1); if (m1 < 1e-9) break;
        const mx = x + (dir * step * 0.5 * u1) / m1, my = y + (dir * step * 0.5 * v1) / m1;
        const [u2, v2] = vecAt(mx, my), m2 = Math.hypot(u2, v2); if (m2 < 1e-9) break;
        x += (dir * step * u2) / m2; y += (dir * step * v2) / m2;
        if (x < 0 || y < 0 || x > n || y > n) break;
        ctx.lineTo(x0 + (x / n) * s, y0 + (1 - y / n) * s);
      }
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawCoil(d, x0, y0, s) {
  const outSign = d.plane === "xz" ? -1 : 1; // screen-out direction is u × v
  ctx.save(); ctx.lineCap = "round";
  for (const [u1, v1, w1, u2, v2, w2, sg] of d.coil) {
    const dist = Math.min(Math.abs(w1 - d.offset), Math.abs(w2 - d.offset));
    const a = Math.max(0.12, 1 - dist / (d.extent * 1.5));
    ctx.strokeStyle = sg >= 0 ? `rgba(92,207,230,${a})` : `rgba(255,120,200,${a})`; ctx.lineWidth = 1.2;
    const [ax, ay] = toPx(u1, v1, d, x0, y0, s), [bx, by] = toPx(u2, v2, d, x0, y0, s);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    if ((w1 - d.offset) * (w2 - d.offset) < 0) { // wire pierces the slice: ⊙ out of page, ⊗ into page
      const t = (d.offset - w1) / (w2 - w1), [cx, cy] = toPx(u1 + (u2 - u1) * t, v1 + (v2 - v1) * t, d, x0, y0, s);
      const out = Math.sign(w2 - w1) * sg * outSign > 0;
      ctx.fillStyle = "#0a0d13"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 7); ctx.fill(); ctx.stroke();
      ctx.beginPath();
      if (out) { ctx.arc(cx, cy, 1.6, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); }
      else { ctx.moveTo(cx - 3, cy - 3); ctx.lineTo(cx + 3, cy + 3); ctx.moveTo(cx + 3, cy - 3); ctx.lineTo(cx - 3, cy + 3); ctx.stroke(); }
    }
  }
  ctx.restore();
}

function drawAxes(d, x0, y0, s) {
  const [lu, lv] = { xy: ["x", "y"], xz: ["x", "z"], yz: ["y", "z"] }[d.plane];
  ctx.fillStyle = "#8b97ab"; ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.textAlign = "center"; ctx.fillText(`${lu}  ±${(d.extent * 100).toFixed(0)} cm`, x0 + s / 2, y0 + s + 14);
  ctx.save(); ctx.translate(x0 + 12, y0 + s / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`${lv}  ±${(d.extent * 100).toFixed(0)} cm`, 0, 0); ctx.restore();
}

function drawLegend(vmin, vmax) {
  const bar = $("bar").getContext("2d");
  for (let i = 0; i < 180; i++) { const [r, g, b] = cmap(i / 179); bar.fillStyle = `rgb(${r},${g},${b})`; bar.fillRect(i, 0, 1, 10); }
  $("lg-min").textContent = $("log").checked ? fmtT(vmin) : "0";
  $("lg-max").textContent = fmtT(vmax);
}

view.addEventListener("mousemove", (e) => {
  const d = state.data; if (!d || !geom) return;
  const r = view.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const fx = (px - geom.x0) / geom.side, fy = (py - geom.y0) / geom.side;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) { $("probe").textContent = "Hover the field"; return; }
  const n = d.res, fi = fx * n - 0.5, fj = (1 - fy) * n - 0.5;
  const u = (fx * 2 - 1) * d.extent, v = ((1 - fy) * 2 - 1) * d.extent;
  const [lu, lv] = { xy: ["x", "y"], xz: ["x", "z"], yz: ["y", "z"] }[d.plane];
  $("probe").textContent = `${lu}=${(u * 100).toFixed(1)} ${lv}=${(v * 100).toFixed(1)} cm  |B|=${fmtT(sample(d.bmag, n, fi, fj))}  B${lu}=${fmtT(sample(d.bu, n, fi, fj))}  B${lv}=${fmtT(sample(d.bv, n, fi, fj))}`;
});
view.addEventListener("mouseleave", () => ($("probe").textContent = "Hover the field"));
new ResizeObserver(() => draw()).observe(view);

// ---------- benchmark tab ----------
const SERIES = [
  ["cpu1", "CPU 1 thread f64", "#6b7a90"], ["cpu64", "CPU Rayon f64", "#ffb454"], ["cpu32", "CPU Rayon f32", "#ff7a59"],
  ["gpu_naive", "GPU naive", "#8f7bff"], ["gpu_tiled", "GPU tiled (total)", "#5ccfe6"], ["gpu_kernel", "GPU tiled (kernel only)", "#7fd962"],
];
function drawBench(res) {
  const svg = $("b-chart"), W = 760, H = 360, m = { l: 62, r: 34, t: 14, b: 40 };
  const xs = res.rows.map((r) => r.n);
  const all = res.rows.flatMap((r) => SERIES.map(([k]) => r[k]).filter((v) => v != null && v > 0));
  const lx = (v) => Math.log10(v), minX = lx(Math.min(...xs)), maxX = lx(Math.max(...xs));
  const minY = Math.floor(Math.log10(Math.min(...all))), maxY = Math.ceil(Math.log10(Math.max(...all)));
  const X = (v) => m.l + ((lx(v) - minX) / Math.max(1e-9, maxX - minX)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((Math.log10(v) - minY) / Math.max(1, maxY - minY)) * (H - m.t - m.b);
  let s = "";
  for (let p = minY; p <= maxY; p++) { const y = Y(10 ** p); s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="#232c3b"/><text x="${m.l - 8}" y="${y + 4}" text-anchor="end">${10 ** p >= 1 ? 10 ** p : (10 ** p).toFixed(-p)} ms</text>`; }
  xs.forEach((n) => { const x = X(n); s += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H - m.b}" stroke="#1a2230"/><text x="${x}" y="${H - m.b + 16}" text-anchor="middle">${n.toLocaleString()}</text>`; });
  s += `<text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">number of filaments</text>`;
  if (res.crossover) { const x = X(res.crossover); s += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H - m.b}" stroke="#5ccfe6" stroke-dasharray="4 4"/><text x="${x + 6}" y="${m.t + 12}" fill="#5ccfe6" style="fill:#5ccfe6">GPU ahead from here</text>`; }
  const legend = [];
  for (const [k, label, color] of SERIES) {
    const pts = res.rows.filter((r) => r[k] != null && r[k] > 0).map((r) => [X(r.n), Y(r[k])]);
    if (!pts.length) continue;
    s += `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
    s += pts.map((p) => `<circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${color}"/>`).join("");
    legend.push(`<span><i style="background:${color}"></i>${label}</span>`);
  }
  svg.innerHTML = s; $("b-legend").innerHTML = legend.join("");
}

async function runBench() {
  const btn = $("b-run"); btn.disabled = true; btn.textContent = "Running…";
  try {
    const sizes = $("b-sizes").value.split(/[ ,]+/).map(Number).filter((n) => n > 0);
    const res = await api("/api/bench", { points: +$("b-points").value, sizes, repeats: 3 });
    drawBench(res);
    const best = res.rows.filter((r) => r.speedup != null).sort((a, b) => b.speedup - a.speedup)[0];
    const bestK = res.rows.filter((r) => r.kernel_speedup != null).sort((a, b) => b.kernel_speedup - a.kernel_speedup)[0];
    const hasGpu = !!state.info?.gpu;
    $("b-kpis").innerHTML = [
      [res.crossover ? res.crossover.toLocaleString() : hasGpu ? "none" : "no GPU", "filaments where GPU overtakes CPU (end to end)"],
      [best ? best.speedup.toFixed(1) + "×" : "–", best ? `best end-to-end speedup (${best.n.toLocaleString()} filaments)` : "best speedup"],
      [bestK ? bestK.kernel_speedup.toFixed(1) + "×" : "–", "best kernel-only speedup vs Rayon f64"],
      [res.points.toLocaleString(), "field points per run"],
    ].map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
    $("b-table").innerHTML = `<thead><tr><th>filaments</th><th>CPU 1 thr</th><th>CPU f64</th><th>CPU f32</th><th>GPU naive</th><th>GPU tiled</th><th>GPU kernel</th><th>speedup</th></tr></thead><tbody>` +
      res.rows.map((r) => `<tr><td>${r.n.toLocaleString()}</td><td>${fmtMs(r.cpu1)}</td><td>${fmtMs(r.cpu64)}</td><td>${fmtMs(r.cpu32)}</td><td>${fmtMs(r.gpu_naive)}</td><td>${fmtMs(r.gpu_tiled)}</td><td>${fmtMs(r.gpu_kernel)}</td><td>${r.speedup == null ? "–" : r.speedup.toFixed(2) + "×"}</td></tr>`).join("") + "</tbody>";
    if (res.tune.length) {
      const max = Math.max(...res.tune.map((t) => t.ms)), min = Math.min(...res.tune.map((t) => t.ms));
      $("b-tune-note").textContent = `Kernel-only time at ${res.tune_filaments.toLocaleString()} filaments. Shorter is faster.`;
      $("b-tune").innerHTML = res.tune.map((t) => `<div class="bar-row"><span>${t.kernel} · wg ${t.wg}</span><div><div class="bar ${t.ms === min ? "best" : ""}" style="width:${(t.ms / max) * 100}%"></div></div><span>${fmtMs(t.ms)} ms</span></div>`).join("");
    } else { $("b-tune-note").textContent = "No GPU detected, so there is nothing to tune."; $("b-tune").innerHTML = ""; }
  } catch (e) { $("b-kpis").innerHTML = `<div class="kpi"><div class="l" style="color:var(--bad)">${e.message}</div></div>`; }
  btn.disabled = false; btn.textContent = "Run benchmark";
}

async function runValidate() {
  const btn = $("v-run"); btn.disabled = true; btn.textContent = "Running…";
  try {
    const r = await api("/api/validate", {});
    $("v-table").innerHTML = `<thead><tr><th>check</th><th>cases</th><th>max rel. error</th><th>tolerance</th><th>result</th></tr></thead><tbody>` +
      r.checks.map((c) => `<tr><td class="name">${c.name}<div class="muted" style="font-size:11px">${c.note}</div></td><td>${c.cases.toLocaleString()}</td><td>${sci(c.max_rel_err)}</td><td>${sci(c.tolerance)}</td><td><span class="badge ${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</span></td></tr>`).join("") + "</tbody>";
  } catch (e) { $("v-table").innerHTML = `<tbody><tr><td style="color:var(--bad)">${e.message}</td></tr></tbody>`; }
  btn.disabled = false; btn.textContent = "Run validation";
}

// ---------- tabs + init ----------
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach((c) => c.setAttribute("aria-selected", c === b));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + b.dataset.tab));
  if (b.dataset.tab === "field") draw();
}));

async function init() {
  for (const [id, v] of Object.entries({ radius: 0.1, current: 10, turns: 1, length: 0.4, per_turn: 128, offset: 0, extent: 0.25 })) $(id).value = v;
  buildPresets(); syncLabels();
  document.querySelectorAll(".panel input[type=range], #res, #backend").forEach((el) => el.addEventListener("input", () => { syncLabels(); refreshSoon(); }));
  ["log", "lines", "coil"].forEach((id) => $(id).addEventListener("change", draw));
  $("b-run").onclick = runBench; $("v-run").onclick = runValidate;
  try {
    const info = await api("/api/info"); state.info = info;
    $("chip-cpu").textContent = `CPU: ${info.cpu_threads} threads`;
    const g = $("chip-gpu");
    if (info.gpu) { g.textContent = `GPU: ${info.gpu.name} · ${info.gpu.backend}`; g.classList.add("on"); $("backend").value = "gpu-tiled"; }
    else { g.textContent = "GPU: unavailable"; document.querySelectorAll('#backend option[value^="gpu"]').forEach((o) => (o.disabled = true)); }
  } catch (e) { $("status").textContent = "Cannot reach the FieldKernels server."; return; }
  refresh();
}
init();
