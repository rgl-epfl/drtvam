import mitsuba as mi
import drjit as dr
import pytest

import drtvam
from drtvam.convolution import make_drjit_conv
import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_mitsuba_convolution(variant):
    mi.set_variant(variant)
    # Parameters
    spacing = 5e-3
    D = 10e-6
    dt = 10
    radius = 20
    size = 40   # 21

    # Build kernel
    conv = make_drjit_conv(spacing * 3, spacing * 1, spacing * 2, D, dt,
                           radiusz=radius, radiusx=radius, radiusy=radius)

    # Input: 3D array with a single 1 at the center
    arr = dr.reshape(mi.TensorXf(dr.zeros(mi.Float, size * size *  size)), (size, size, size, 1))
    arr[size//2, size//2, size//2] = 1.0
    r = arr

    # Convolve
    result = conv(r)
    result_np = (result)

    # plt.imshow(np.sum(result, axis=3)[:, :, size//2])
    # plt.colorbar()
    # plt.show()

    # Build expected kernel analytically (separable outer product)
    idx = mi.TensorXf(dr.arange(mi.Float, -radius, radius + 1) * spacing)[:-1]
    idy = mi.TensorXf(dr.arange(mi.Float, -radius, radius + 1) * spacing * 2)[:-1]
    idz = mi.TensorXf(dr.arange(mi.Float, -radius, radius + 1) * spacing * 3)[:-1]
    k1x = dr.exp(-(idx**2 / (4 * D * dt)))
    k1x /= dr.sum(k1x)
    k1y = dr.exp(-(idy**2 / (4 * D * dt)))
    k1y /= dr.sum(k1y)
    k1z = dr.exp(-(idz**2 / (4 * D * dt)))
    k1z /= dr.sum(k1z)

    print(k1z)
    kernel_3d = k1z[:, None, None, None] * k1x[None, :, None, None] * k1y[None, None, :, None]
    kernel_3d /= dr.sum(kernel_3d)

    # plt.imshow(np.sum(kernel_3d, axis=3)[:, :, size//2])
    # plt.colorbar()
    # plt.show()

    # Compare
    max_err = dr.max(dr.abs(result_np - kernel_3d))
    assert dr.abs(dr.sum(result) - 1.0) < 1e-6, "Convolution result is not normalized!"
    assert dr.abs(dr.sum(kernel_3d) - 1.0) < 1e-6, "Expected kernel is not normalized!"
    assert max_err < 1e-6, "Convolution result does not match expected kernel!"
    assert dr.allclose(result_np, kernel_3d, atol=1e-6), "Convolution result does not match expected kernel!"


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_convolution_against_scipy_fft(variant):
    mi.set_variant(variant)
    sfft = pytest.importorskip("scipy.fft")

    scalex, scaley, scalez = 1e-3, 1e-3, 1e-3
    resx, resy, resz = 90, 90, 90

    dx, dy, dz = scalex / resx, scaley / resy, scalez / resz
    delta_t = 1
    D = 1e-3
    filter_radius = 45

    z_ax = np.arange(resz, dtype=np.float32) * dz - scalez / 2
    x_ax = np.arange(resx, dtype=np.float32) * dx - scalex / 2
    y_ax = np.arange(resy, dtype=np.float32) * dy - scaley / 2
    Z, X, Y = np.meshgrid(z_ax, x_ax, y_ax, indexing='ij')

    limit  = 0.01e-3
    vol_np = ((np.abs(Z) <= limit) & (np.abs(Y) <= limit) & (np.abs(X) <= limit)).astype(np.float32)

    sigma = np.sqrt(2 * delta_t * D)
    kernel_3d  = np.exp(-(Z**2 + X**2 + Y**2) / (2 * sigma**2)).astype(np.float32)
    kernel_3d /= kernel_3d.sum()
    kernel_3d  = sfft.ifftshift(kernel_3d)

    def fft_convolve_3d(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        vol_f = sfft.rfftn(volume, axes=(0, 1, 2))
        ker_f = sfft.rfftn(kernel, s=volume.shape, axes=(0, 1, 2))
        return sfft.irfftn(vol_f * ker_f, s=volume.shape, axes=(0, 1, 2))

    result_fft_np = fft_convolve_3d(vol_np, kernel_3d)

    vol_drjit = mi.TensorXf(vol_np)
    drjit_conv = make_drjit_conv(dx, dy, dz, D, delta_t, filter_radius, filter_radius, filter_radius)
    result_drjit = drjit_conv(vol_drjit)
    result_drjit_np = result_drjit.numpy().reshape(resz, resx, resy)

    assert np.allclose(result_drjit_np, result_fft_np, atol=1e-6), "Convolution result does not match expected result from scipy fft!"


