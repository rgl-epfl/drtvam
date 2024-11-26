import mitsuba as mi

def integrator_variant_callback(old, new):
    import importlib
    importlib.reload(mi.ad.integrators.common)

    from . import common
    importlib.reload(common)

    from . import radon
    importlib.reload(radon)

    from . import volume
    importlib.reload(volume)

mi.detail.add_variant_callback(integrator_variant_callback)