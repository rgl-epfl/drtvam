import pytest
import mitsuba as mi
import drjit as dr
import drtvam

@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_crop(variant):
    mi.set_variant(variant)
    projector = mi.load_dict({
        "type": "collimated",
        "n_patterns": 1,
        "resx": 20,
        "resy": 10,
        "cropx": 4,
        "cropy": 4,
        "crop_offset_x": 8,
        "crop_offset_y": 3,
        "pixel_size": 1.,
        "motion": "circular",
        "distance": 20
    })

    integrator = mi.load_dict({
        "type": "volumeintegrator",
        "print_time": 1.
    })
    sampler = mi.load_dict({'type': 'independent'})

    spp = 128
    sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp)
    wavefront_size = projector.active_size() * spp
    sampler.seed(0, wavefront_size)

    t = mi.Float(0)
    ray, weight, _ = integrator.sample_rays(None, projector, sampler)

    assert dr.all((ray.o.y > -2) & (ray.o.y < 2) & (ray.o.z > -2) & (ray.o.z < 2) , axis=None)

