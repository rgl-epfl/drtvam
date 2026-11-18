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

        # Order in which rays are traced: 'pattern' (pattern by pattern), 'slice' (DMD row by
        # row, all patterns of a row together) or 'auto' ('slice' for dose volumes larger than
        # ray_order_min_mb). With a collimated projector each DMD row lights one z-slice, so in
        # slice order the rays in flight write to a few slices only, which stay in the GPU cache.
        self.ray_order = props.get('ray_order', 'auto')
        if self.ray_order not in ('auto', 'pattern', 'slice'):
            raise ValueError(f"[{self.__class__.__name__}] ray_order must be 'auto', 'pattern' or 'slice', "
                             f"got '{self.ray_order}'.")
        self.ray_order_min_mb = props.get('ray_order_min_mb', 64.)

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

    def use_slice_order(self, sensor: mi.Sensor) -> bool:
        if self.ray_order != 'auto':
            return self.ray_order == 'slice'
        n_bytes = 4 * dr.prod(sensor.film().data.shape)
        return n_bytes > self.ray_order_min_mb * 2**20

    def ray_subset(self, emitter: TVAMProjector, sensor: mi.Sensor, skip_zero: bool):
        """
        Indices into the active pixel arrays of the pixels to trace, in tracing order,
        or None to trace all active pixels in their stored order.
        """
        slice_order = self.use_slice_order(sensor)
        if not (skip_zero or slice_order):
            return None

        if skip_zero:
            # A zero pixel deposits no dose
            subset = dr.compress(emitter.active_data != 0.)
        else:
            subset = dr.arange(mi.UInt32, emitter.active_size())

        if slice_order and dr.width(subset) > 0:
            # Sort by (row, pattern, column): scatter each index to its key, then read
            # the occupied keys back in increasing order
            n_patterns, h, w = emitter.size()
            pixel = dr.gather(mi.UInt32, emitter.active_pixels, subset)
            pattern, row, col = pixel // (h * w), (pixel // w) % h, pixel % w
            key = (row * n_patterns + pattern) * w + col
            sorted_idx = dr.zeros(mi.UInt32, n_patterns * h * w)
            dr.scatter(sorted_idx, subset + 1, key)
            subset = dr.gather(mi.UInt32, sorted_idx, dr.compress(sorted_idx != 0)) - 1

        return subset

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

