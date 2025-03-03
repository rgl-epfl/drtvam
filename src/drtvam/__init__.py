import mitsuba as mi
if ('llvm_ad_mono' not in mi.variants()) and ('cuda_ad_mono' not in mi.variants()):
    raise ImportError("This package requires Mitsuba's'llvm_ad_mono' or 'cuda_ad_mono' variants to be available.")
import drjit as dr

#dr.set_thread_count(2)

def plugin_variant_callback(old, new):
    if new not in ['cuda_ad_mono', 'llvm_ad_mono']:
        raise ValueError(f"Unsupported variant '{new}', it must be either 'cuda_ad_mono' or 'llvm_ad_mono'.")

    import importlib

    from . import projector
    importlib.reload(projector)

    from . import film
    importlib.reload(film)

    from . import sensor
    importlib.reload(sensor)

mi.detail.add_variant_callback(plugin_variant_callback)
from . import integrators

mi.set_variant('llvm_ad_mono')

from . import geometry, motion, loss

def register_geometry(name, cls):
    if name in geometry.geometries:
        raise ValueError(f"Geometry '{name}' is already registered.")
    if not issubclass(cls, geometry.Container):
        raise ValueError(f"Class '{cls}' is not a subclass of 'geometry.Container'.")
    geometry.geometries[name] = cls

def register_motion(name, cls):
    if name in motion.motions:
        raise ValueError(f"Motion '{name}' is already registered.")
    if not issubclass(cls, motion.Motion):
        raise ValueError(f"Class '{cls}' is not a subclass of 'motion.Motion'.")
    motion.motions[name] = cls

def register_loss(name, cls):
    if name in loss.losses:
        raise ValueError(f"Loss '{name}' is already registered.")
    if not issubclass(cls, loss.Loss):
        raise ValueError(f"Class '{cls}' is not a subclass of 'loss.Loss'.")
    loss.losses[name] = cls

