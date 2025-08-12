import pytest
import drtvam
import mitsuba as mi
import drjit as dr

#TODO: test with analytic GT
def build_scene(method):
    d_ext = 16.77
    d_int = 15.33
    scene_dict = {
        'type': 'scene',
        'dmd': {
            'type': 'collimated',
            'patterns': mi.TensorXf(dr.linspace(mi.Float, 1, 10, 100*128*128), (100, 128, 128)),
            'pixel_size': d_ext/128,
            'motion': 'circular',
            'distance': 1.5 * d_ext,
        },
        'sensor': {
            'type': method,
            'to_world': mi.ScalarTransform4f().scale(d_ext),
            'film': {
                'type': 'vfilm',
                'resx': 128,
                'resy': 128,
                'resz': 128
            }
        },
        'integrator': { 
            'type': 'volumeintegrator',
            'print_time': 1, # To match previous implementation
            'max_depth': 32,
            'rr_depth': 3,
            'sample_time': True,
        },
        'vial_exterior': {
            'type': 'cylinder',
            'p0': [0., 0., -10.],
            'p1': [0., 0.,  10.],
            'radius': 0.5 * d_ext,
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.514
            }
        },
        'vial_interior': {
            'type': 'cylinder',
            'p0': [0., 0., -10.],
            'p1': [0., 0.,  10.],
            'radius': 0.5 * d_int,
            'bsdf': {
                'type': 'dielectric',
                'ext_ior': 1.514,
                'int_ior': 1.4849
            },
            'interior': {
                'type': 'homogeneous',
                'sigma_t': 0.1,
                'albedo': 0.5,
                'phase': {'type': 'rayleigh'}
            }
        },
    }
    if method == 'ratio':
        scene_dict['sensor']['majorant'] = 10.

    return mi.load_dict(scene_dict)

@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_reverse_ad(variant):
    mi.set_variant(variant)
    scene = build_scene('dda')
    params = mi.traverse(scene)
    param_key = 'dmd.active_data'
    patterns = mi.TensorXf(params[param_key])

    _a = 1.0
    eps = 1e-3
    spp = 128

    # FD gradient
    a = mi.Float(_a)
    params[param_key] = patterns * (a + eps)
    params.update()
    vol1 = mi.render(scene, params, spp=spp, seed=0)
    loss1 = dr.mean(dr.square(vol1), axis=None)

    params[param_key] = patterns * (a - eps)
    params.update()
    vol2 = mi.render(scene, params, spp=spp, seed=0)
    loss2 = dr.mean(dr.square(vol2), axis=None)

    fd_grad = (loss1 - loss2) / (2 * eps)

    # AD gradient
    for method in ('dda', 'ratio', 'delta'):
        scene = build_scene(method)
        params = mi.traverse(scene)
        a = dr.opaque(mi.Float, _a)

        dr.enable_grad(a)
        params[param_key] = a * patterns
        params.update()

        vol = mi.render(scene, params, spp=spp)

        loss = dr.mean(dr.square(vol), axis=None)
        dr.backward(loss)

        assert dr.abs((a.grad - fd_grad) / fd_grad) < 2e-4

