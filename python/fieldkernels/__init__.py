"""FieldKernels: Biot-Savart fields of straight current filaments.

    >>> import numpy as np, fieldkernels as fk
    >>> starts, ends, I = fk.loop_filaments([0, 0, 0], [0, 0, 1], radius=0.1, n=256, current=10.0)
    >>> B = fk.biot_savart(np.array([[0.0, 0.0, 0.05]]), starts, ends, I)   # tesla, shape (1, 3)
"""
from __future__ import annotations

import numpy as np

try:
    from . import _native
except ImportError:  # keep the package importable (NumPy fallback) if the extension is missing
    _native = None

__all__ = ["biot_savart", "biot_savart_numpy", "loop_filaments", "gpu_info", "MU0"]

MU0 = 4e-7 * np.pi


def _prep(points, starts, ends, currents):
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    starts = np.ascontiguousarray(starts, dtype=np.float64).reshape(-1, 3)
    ends = np.ascontiguousarray(ends, dtype=np.float64).reshape(-1, 3)
    currents = np.ascontiguousarray(np.broadcast_to(np.asarray(currents, dtype=np.float64), (len(starts),)))
    if not (len(starts) == len(ends) == len(currents)):
        raise ValueError("starts, ends and currents must have the same length")
    return points, starts, ends, currents


def gpu_info():
    """Name of the usable GPU adapter, or None."""
    return _native.gpu_info() if _native else None


def loop_filaments(center, normal, radius, n=256, current=1.0):
    """Circular loop as (starts, ends, currents). Current is counter-clockwise about `normal`."""
    if _native:
        s, e, i = _native.loop_filaments(list(map(float, center)), list(map(float, normal)), radius, n, current)
        return s, e, np.asarray(i)
    nrm = np.asarray(normal, float) / np.linalg.norm(normal)
    h = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(h, nrm)
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.asarray(center, float) + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    return ring, np.roll(ring, -1, axis=0), np.full(n, float(current))


def biot_savart_numpy(points, starts, ends, currents, chunk=256):
    """Reference NumPy implementation in float64 (memory ~ chunk x filaments)."""
    points, starts, ends, currents = _prep(points, starts, ends, currents)
    dl = ends - starts
    l2 = np.einsum("ij,ij->i", dl, dl)
    out = np.empty_like(points)
    for k in range(0, len(points), chunk):
        p = points[k:k + chunk, None, :]
        r1, r2 = p - starts, p - ends
        c = np.cross(dl, r1)
        c2 = np.einsum("pij,pij->pi", c, c)
        ok = (l2 > 0) & (c2 > 1e-24 * l2 * l2)
        t = (np.einsum("ij,pij->pi", dl, r1) / np.linalg.norm(r1, axis=2)
             - np.einsum("ij,pij->pi", dl, r2) / np.linalg.norm(r2, axis=2))
        w = np.where(ok, currents * t / np.where(ok, c2, 1.0), 0.0)
        out[k:k + chunk] = 1e-7 * np.einsum("pi,pij->pj", w, c)
    return out


def biot_savart(points, starts, ends, currents, backend="auto", precision="f64", **kw):
    """Magnetic field in tesla at `points` (N, 3) from straight filaments.

    backend: "auto" (CPU f64 unless precision="f32" and a GPU exists), "gpu", "cpu", "numpy".
    precision: "f64" or "f32" (CPU only; the GPU path is always f32).
    Extra kwargs go to the GPU call: kernel="tiled"|"naive", workgroup_size=int.
    """
    points, starts, ends, currents = _prep(points, starts, ends, currents)
    if backend == "numpy" or _native is None:
        return biot_savart_numpy(points, starts, ends, currents)
    if backend == "auto":
        backend = "gpu" if (precision == "f32" and _native.gpu_info()) else "cpu"
    if backend == "gpu":
        return _native.biot_savart_gpu(points, starts, ends, currents, **kw)[0]
    if backend == "cpu":
        return _native.biot_savart_cpu(points, starts, ends, currents, precision, True)
    raise ValueError(f"unknown backend {backend!r}")
