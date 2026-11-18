import numpy as np
import pytest
import drtvam
import mitsuba as mi
import drjit as dr

D_EXT, D_INT, RES, N_PAT = 16.77, 15.33, 32, 16


def build_scene(albedo=0.0, sigma_t=0.1, res=RES):
    rng = np.random.default_rng(1)
    p = rng.random((N_PAT, res, res)).astype(np.float32)
    p[p < 0.6] = 0.  # sparse: ~60 % of the pixels are off
    return mi.load_dict({
        'type': 'scene',
        'dmd': {'type': 'collimated', 'patterns': mi.TensorXf(p), 'pixel_size': D_EXT / res,
                'motion': 'circular', 'distance': 1.5 * D_EXT},
        'sensor': {'type': 'dda', 'to_world': mi.ScalarTransform4f().scale(D_EXT),
                   'film': {'type': 'vfilm', 'resx': res, 'resy': res, 'resz': res}},
        'vial_exterior': {'type': 'cylinder', 'p0': [0., 0., -10.], 'p1': [0., 0., 10.], 'radius': 0.5 * D_EXT,
                          'bsdf': {'type': 'dielectric', 'int_ior': 1.514}},
        'vial_interior': {'type': 'cylinder', 'p0': [0., 0., -10.], 'p1': [0., 0., 10.], 'radius': 0.5 * D_INT,
                          'bsdf': {'type': 'dielectric', 'ext_ior': 1.514, 'int_ior': 1.4849},
                          'interior': {'type': 'homogeneous', 'sigma_t': sigma_t, 'albedo': albedo,
                                       'phase': {'type': 'isotropic'}}},
    })


def integrator(**kw):
    return mi.load_dict({'type': 'volumeintegrator', 'transmission_only': True, 'max_depth': 16} | kw)


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_skip_zero_rays_is_exact(variant):
    """Deterministic rays: leaving out zero pixels must not change the dose."""
    mi.set_variant(variant)
    scene = build_scene()
    ref = mi.render(scene, integrator=integrator(regular_sampling=True, skip_zero_rays=False), spp=1).numpy()
    out = mi.render(scene, integrator=integrator(regular_sampling=True, skip_zero_rays=True), spp=1).numpy()
    assert ref.max() > 0
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-6 * ref.max())


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_skip_zero_rays_all_zero(variant):
    mi.set_variant(variant)
    scene = build_scene()
    params = mi.traverse(scene)
    params['dmd.active_data'] = dr.zeros(mi.Float, dr.width(params['dmd.active_data']))
    params.update()
    out = mi.render(scene, params, integrator=integrator(regular_sampling=True, skip_zero_rays=True), spp=1)
    assert out.shape == (RES, RES, RES, 1)
    assert dr.all(out.array == 0.)


@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_skip_zero_rays_keeps_gradients(variant):
    """Only the primal pass skips rays: every pixel, including the dark ones, still gets its gradient."""
    mi.set_variant(variant)
    scene = build_scene()
    params = mi.traverse(scene)
    key = 'dmd.active_data'
    p0 = mi.Float(params[key])
    w = mi.Float(np.random.default_rng(2).random(RES ** 3).astype(np.float32))
    grads = []
    for skip in (False, True):
        params[key] = mi.Float(p0)
        dr.enable_grad(params[key])
        params.update()
        vol = mi.render(scene, params, integrator=integrator(regular_sampling=True, skip_zero_rays=skip),
                        spp=1, seed=0, seed_grad=1)
        dr.backward(dr.sum(vol.array * w))
        grads.append(dr.grad(params[key]).numpy())
    zero = p0.numpy() == 0
    assert np.allclose(grads[1], grads[0], rtol=1e-5, atol=1e-6 * np.abs(grads[0]).max())
    assert np.abs(grads[1][zero]).max() > 0


@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
def test_skip_zero_rays_monte_carlo(variant):
    """Random sub-pixel rays and scattering: same expected dose (different random streams)."""
    mi.set_variant(variant)
    scene = build_scene(albedo=0.5)
    a = mi.render(scene, integrator=integrator(skip_zero_rays=False), spp=64, seed=3).numpy()
    b = mi.render(scene, integrator=integrator(skip_zero_rays=True), spp=64, seed=4).numpy()
    assert abs(b.sum() / a.sum() - 1) < 0.01
