import numpy as np
import pytest
import drtvam
import mitsuba as mi
import drjit as dr
from test_skip_zero_rays import build_scene, integrator, RES, N_PAT


def render_with_grad(scene, params, integ, w):
    key = 'dmd.active_data'
    dr.enable_grad(params[key])
    params.update()
    vol = mi.render(scene, params, integrator=integ, spp=1, seed=0, seed_grad=1)
    dr.backward(dr.sum(vol.array * w))
    g = dr.grad(params[key]).numpy()
    dr.disable_grad(params[key])
    return vol.numpy(), g


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
@pytest.mark.parametrize("active_subset", [False, True])
@pytest.mark.parametrize("skip_zero", [False, True])
def test_slice_order_is_exact(variant, active_subset, skip_zero):
    """Slice order gives the same dose and gradients as pattern order, also when only
    a subset of the pixels is active (as after filter_radon)."""
    mi.set_variant(variant)
    scene = build_scene()
    params = mi.traverse(scene)
    if active_subset:
        keep = np.random.default_rng(3).random(N_PAT * RES * RES) < 0.5
        params['dmd.active_pixels'] = mi.UInt32(np.nonzero(keep)[0].astype(np.uint32))
        params['dmd.active_data'] = mi.Float(params['dmd.active_data'].numpy()[keep])
        params.update()
    p0 = mi.Float(params['dmd.active_data'])
    w = mi.Float(np.random.default_rng(2).random(RES ** 3).astype(np.float32))
    out = {}
    for order in ('pattern', 'slice'):
        params['dmd.active_data'] = mi.Float(p0)
        params.update()
        out[order] = render_with_grad(scene, params, integrator(regular_sampling=True, ray_order=order,
                                                                skip_zero_rays=skip_zero), w)
    (va, ga), (vb, gb) = out['pattern'], out['slice']
    assert va.max() > 0
    assert np.allclose(vb, va, rtol=1e-5, atol=1e-6 * va.max())
    assert np.allclose(gb, ga, rtol=1e-5, atol=1e-6 * np.abs(ga).max())


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
@pytest.mark.parametrize("skip_zero", [False, True])
def test_slice_order_permutation(variant, skip_zero):
    """The traced pixels are sorted by (row, pattern, column) and each one appears exactly once."""
    mi.set_variant(variant)
    scene = build_scene()
    proj, sensor = scene.emitters()[0], scene.sensors()[0]
    subset = integrator(ray_order='slice').ray_subset(proj, sensor, skip_zero).numpy()
    data = proj.active_data.numpy()
    expected = np.nonzero(data != 0)[0] if skip_zero else np.arange(data.size)
    assert np.array_equal(np.sort(subset), expected)
    pixel = proj.active_pixels.numpy()[subset]
    n_pat, h, w = proj.size()
    key = ((pixel // w) % h * n_pat + pixel // (h * w)) * w + pixel % w
    assert np.all(np.diff(key) > 0)


@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
def test_ray_order_auto(variant):
    mi.set_variant(variant)
    sensor = build_scene().sensors()[0]
    assert not integrator(ray_order='auto').use_slice_order(sensor)           # 128 KB volume
    assert integrator(ray_order='auto', ray_order_min_mb=0.01).use_slice_order(sensor)
    assert integrator(ray_order='slice').use_slice_order(sensor)
    assert not integrator(ray_order='pattern', ray_order_min_mb=0.).use_slice_order(sensor)
    with pytest.raises((ValueError, RuntimeError), match="ray_order"):
        integrator(ray_order='diagonal')
