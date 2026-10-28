// Biot-Savart kernels (f32). `__WG__` is replaced with the workgroup size at
// pipeline-creation time. Two entry points:
//   naive : every thread streams all filaments from global memory
//   tiled : a workgroup cooperatively stages WG filaments into workgroup
//           (shared) memory, so each global load is reused by WG points.

struct Seg {
    a: vec4<f32>, // xyz = start, w = current
    b: vec4<f32>, // xyz = end
};

struct Params {
    n_pts: u32,
    n_segs: u32,
    stride: u32, // threads per row of a 2-D dispatch (dispatch_x * WG)
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;
@group(0) @binding(2) var<storage, read_write> outb: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = __WG__u;
const EPS: f32 = 1e-8;
const MU0_OVER_4PI: f32 = 1e-7;

fn contrib(p: vec3<f32>, s: Seg) -> vec3<f32> {
    let a = s.a.xyz;
    let b = s.b.xyz;
    let dl = b - a;
    let l2 = dot(dl, dl);
    let r1 = p - a;
    let r2 = p - b;
    let c = cross(dl, r1);
    let c2 = dot(c, c);
    if (l2 <= 0.0 || c2 <= EPS * l2 * l2) {
        return vec3<f32>(0.0);
    }
    let t = dot(dl, r1) * inverseSqrt(dot(r1, r1)) - dot(dl, r2) * inverseSqrt(dot(r2, r2));
    return (s.a.w * t / c2) * c;
}

@compute @workgroup_size(WG)
fn naive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.stride + gid.x;
    if (i >= params.n_pts) {
        return;
    }
    let p = pts[i].xyz;
    var total = vec3<f32>(0.0);
    var j = 0u;
    loop {
        if (j >= params.n_segs) { break; }
        let end = min(j + 64u, params.n_segs);
        var part = vec3<f32>(0.0);
        for (var k = j; k < end; k++) {
            part += contrib(p, segs[k]);
        }
        total += part; // blocked summation: error grows ~sqrt(N/64) not N
        j = end;
    }
    outb[i] = vec4<f32>(total * MU0_OVER_4PI, 0.0);
}

var<workgroup> tile: array<Seg, WG>;

@compute @workgroup_size(WG)
fn tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = gid.y * params.stride + gid.x;
    let valid = i < params.n_pts;
    // Out-of-range threads still take part in the barriers; give them a safe point.
    var p = vec3<f32>(0.0);
    if (valid) {
        p = pts[i].xyz;
    }
    var total = vec3<f32>(0.0);
    var base = 0u;
    loop {
        if (base >= params.n_segs) { break; }
        let idx = base + li;
        if (idx < params.n_segs) {
            tile[li] = segs[idx];
        }
        workgroupBarrier();
        let cnt = min(WG, params.n_segs - base);
        var part = vec3<f32>(0.0);
        for (var k = 0u; k < cnt; k++) {
            part += contrib(p, tile[k]);
        }
        total += part; // per-tile partial sums
        workgroupBarrier();
        base += WG;
    }
    if (valid) {
        outb[i] = vec4<f32>(total * MU0_OVER_4PI, 0.0);
    }
}
