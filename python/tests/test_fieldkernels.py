import numpy as np
import pytest

import fieldkernels as fk

MU0 = 4e-7 * np.pi


def loop(n=512):
    return fk.loop_filaments([0, 0, 0], [0, 0, 1], 0.1, n, 10.0)


def rel(a, b):
    return np.max(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1))


def test_loop_center_field():
    s, e, i = loop(4096)
    b = fk.biot_savart(np.array([[0.0, 0.0, 0.0]]), s, e, i, backend="cpu")
    assert b[0, 2] == pytest.approx(MU0 * 10.0 / (2 * 0.1), rel=1e-6)


def test_numpy_reference_matches_native():
    s, e, i = loop()
    pts = np.random.default_rng(0).uniform(-0.2, 0.2, (200, 3))
    np.testing.assert_allclose(
        fk.biot_savart(pts, s, e, i, backend="cpu"), fk.biot_savart_numpy(pts, s, e, i), rtol=1e-9, atol=1e-15)


def test_f32_close_to_f64():
    s, e, i = loop()
    pts = np.random.default_rng(1).uniform(-0.2, 0.2, (200, 3))
    assert rel(fk.biot_savart(pts, s, e, i, backend="cpu", precision="f32"),
               fk.biot_savart(pts, s, e, i, backend="cpu")) < 1e-4


@pytest.mark.skipif(fk.gpu_info() is None, reason="no GPU adapter")
@pytest.mark.parametrize("kernel", ["naive", "tiled"])
def test_gpu_matches_cpu(kernel):
    s, e, i = loop()
    pts = np.random.default_rng(2).uniform(-0.2, 0.2, (500, 3))
    g = fk.biot_savart(pts, s, e, i, backend="gpu", kernel=kernel)
    assert rel(g, fk.biot_savart(pts, s, e, i, backend="cpu")) < 1e-4
