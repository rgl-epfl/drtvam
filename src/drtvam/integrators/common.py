from __future__ import annotations as __annotations__ # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr

class TVAMIntegrator(mi.ad.integrators.common.ADIntegrator):
    def __init__(self, props):
        super().__init__(props)

        # Sample time stochastically ?
        self.sample_time = props.get('sample_time', False)

        # Print time
        self.print_time = props.get('print_time', 1.)

        # ID of the target shape, necessary when surface-aware discretization is enabled
        self.target_id = props.get('target_id', 'target')

        # Always sample only the transmission component at interfaces
        self.transmission_only = props.get('transmission_only', True)

        # Shoot from the center of the pixels only
        self.regular_sampling = props.get('regular_sampling', False)

        # Primal (dose) renders only trace rays of pixels whose value is non-zero:
        # a zero pixel deposits exactly zero dose. Gradient passes always trace all rays.
        self.skip_zero_rays = props.get('skip_zero_rays', False)

    def parse_scene(self, scene: mi.Scene):
        target_shape = None
        medium = None
        for shape in scene.shapes():
            if dr.hint(shape.id() == self.target_id, mode='scalar'):
                target_shape = mi.ShapePtr(shape)

            if dr.hint(shape.interior_medium() is not None, mode='scalar'):
                if medium is not None:
                    raise ValueError("There is more than one medium in the scene. Only one is supported")
                medium = shape.interior_medium()

        if medium is None:
            raise ValueError("No printing medium found in the scene")

        return medium, target_shape

    def prepare(self,
                emitter: TVAMProjector,
                seed: int = 0,
                spp: int = 0,
                n_pixels: int = None):

        original_sampler = emitter.sampler()
        sampler = original_sampler.clone()

        if self.regular_sampling:
            # spp > 1 makes no sense if shooting from the pixel center, so we force it to 1
            spp = 1

        if spp != 0:
            sampler.set_sample_count(spp)
        spp = sampler.sample_count()

        sampler.set_samples_per_wavefront(spp)
        # Number of pixels that will be traced (all active pixels by default)
        if n_pixels is None:
            n_pixels = emitter.active_size()
        wavefront_size = n_pixels * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        return sampler, spp

    def sample_rays(
        self,
        scene: mi.Scene,
        emitter: DMD,
        sampler: mi.Sampler,
        subset: mi.UInt32 = None,
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:

        n_patterns, h, w = emitter.size()
        spp = sampler.sample_count()

        # Compute discrete sample position, optionally only for a subset of the active pixels
        # (given as indices into the active pixel arrays)
        if subset is None:
            idx = dr.repeat(emitter.active_pixels, spp)
            L = dr.repeat(emitter.active_data, spp)
        else:
            idx = dr.repeat(dr.gather(mi.UInt32, emitter.active_pixels, subset), spp)
            L = dr.repeat(dr.gather(mi.Float, emitter.active_data, subset), spp)
        emitter_idx = idx // (h * w)
        pixel_idx = idx % (h * w)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = pixel_idx // w # pixel_idx = row_idx * w + col_idx
        pos.x = dr.fma(mi.UInt32(mi.Int32(-w)), pos.y, pixel_idx) # col_idx = pixel_idx - row_idx * w

        # Cast to floating point and add random offset
        if self.regular_sampling:
            pos_f = mi.Vector2f(pos) + mi.Vector2f(0.5)
        else:
            pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(w, h))
        pos_adjusted = pos_f * scale

        # Time sample, remapped to [0, 1]
        time = mi.Float(emitter_idx)
        if self.sample_time:
            time += sampler.next_1d()
        time /= n_patterns

        with dr.resume_grad():
            ray, weight = emitter.sample_ray(time, mi.Vector2f(0.), pos_adjusted, sampler.next_2d())

        if subset is not None:
            # The projector weight is normalized by the number of traced rays:
            # keep the per-ray weight of a render of all active pixels
            weight *= dr.width(subset) / emitter.active_size()

        # 1/pdf of the time sample
        weight *= self.print_time

        # 1/N
        #weight *= dr.rcp(dr.prod(emitter.size()) * spp))

        return ray, L, weight

