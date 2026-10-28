"""Helmholtz-pair field on a grid, CPU vs GPU. Run after `maturin develop --release`."""
import time

import numpy as np

import fieldkernels as fk

R = 0.1
a = fk.loop_filaments([0, 0, -R / 2], [0, 0, 1], R, 512, 10.0)
b = fk.loop_filaments([0, 0, +R / 2], [0, 0, 1], R, 512, 10.0)
starts, ends, cur = (np.concatenate(x) for x in zip(a, b))
g = np.linspace(-0.05, 0.05, 301)
pts = np.array([[x, 0.0, z] for z in g for x in g])
centre = len(pts) // 2

print("GPU:", fk.gpu_info())
for backend in ("cpu", "gpu"):
    if backend == "gpu" and fk.gpu_info() is None:
        continue
    t = time.perf_counter()
    B = fk.biot_savart(pts, starts, ends, cur, backend=backend)
    print(f"{backend:>4}: {1e3 * (time.perf_counter() - t):8.1f} ms   Bz(centre) = {B[centre, 2] * 1e6:.3f} uT")
print("analytic Helmholtz centre:", (4 / 5) ** 1.5 * 4e-7 * np.pi * 10 / R * 1e6, "uT")
