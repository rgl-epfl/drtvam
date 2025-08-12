from __future__ import annotations as __annotations__ # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr
from drtvam.sensor import DeltaVolumetricSensor
from .common import TVAMIntegrator

class VolumeIntegrator(TVAMIntegrator):

    def to_string(self):
        return ('VolumeIntegrator[\n'
                f'    {self.print_time=},\n'
                f'    {self.transmission_only=},\n'
                f'    {self.sample_time=},\n'
                f'    {self.max_depth=},\n'
                f'    {self.rr_depth=},\n'
                ']')

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if len(scene.emitters()) == 0:
            raise Exception("No projector found in the scene")
        if len(scene.emitters()) > 1:
            raise Exception("The scene contains more than one projector. Only one is supported")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        sensor.film().clear()

        projector = scene.emitters()[0]

        with dr.suspend_grad():
            sampler, spp = self.prepare(projector, seed, spp)
            ray, Le, weight = self.sample_rays(scene, projector, sampler)

            volume = sensor.compute_volume(scene)
            inv_vol = dr.select(volume != 0., dr.rcp(volume), 0.)

            L = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                Le=Le * weight,
                sensor=sensor,
                depth=mi.UInt32(0),
                δL=None,
                active=mi.Bool(True)
            ) * inv_vol

        return L

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if len(scene.emitters()) == 0:
            raise Exception("No projector found in the scene")
        if len(scene.emitters()) > 1:
            raise Exception("The scene contains more than one projector. Only one is supported")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        sensor.film().clear()

        projector = scene.emitters()[0]

        with dr.suspend_grad():
            sampler, spp = self.prepare(projector, seed, spp)
            ray, Le, weight = self.sample_rays(scene, projector, sampler)

            volume = sensor.compute_volume(scene)
            inv_vol = dr.select(volume != 0., dr.rcp(volume), 0.)

            δL = self.sample(
                mode=dr.ADMode.Forward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                Le=Le * weight,
                sensor=sensor,
                depth=mi.UInt32(0),
                δL=None,
                active=mi.Bool(True)
            ) * inv_vol

        return δL

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if len(scene.emitters()) == 0:
            raise Exception("No projector found in the scene")
        if len(scene.emitters()) > 1:
            raise Exception("The scene contains more than one projector. Only one is supported")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]
        sensor.film().clear()

        projector = scene.emitters()[0]

        sampler, spp = self.prepare(projector, seed, spp)
        ray, Le, weight = self.sample_rays(scene, projector, sampler)

        volume = sensor.compute_volume(scene)
        inv_vol = dr.select(volume != 0., dr.rcp(volume), 0.)

        L = self.sample(
            mode=dr.ADMode.Backward,
            scene=scene,
            sampler=sampler,
            ray=ray,
            Le=Le * weight,
            sensor=sensor,
            depth=mi.UInt32(0),
            δL=grad_in * inv_vol,
            active=mi.Bool(True)
        )

        del L

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               Le: mi.Spectrum,
               sensor: mi.Sensor,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float]]:

        medium, target_shape = self.parse_scene(scene)

        is_primal = mode == dr.ADMode.Primal
        is_backward = mode == dr.ADMode.Backward
        is_forward = mode == dr.ADMode.Forward

        sigma_s, _, sigma_t = medium.get_scattering_coefficients(mi.MediumInteraction3f())
        has_scattering = dr.any(dr.any(sigma_s != 0)) # Delta tracking must assume there is scattering to work
        if not has_scattering and isinstance(sensor, DeltaVolumetricSensor):
            raise ValueError("Tried to render a purely absorptive volume with a delta tracking sensor. This is not supported.")

        if is_forward and dr.grad_enabled(Le):
            dr.forward_to(Le)

        attenuation = mi.Spectrum(1.)
        δL = mi.TensorXf(δL) if δL is not None else dr.zeros(mi.TensorXf, sensor.film().resolution())
        em_grad = dr.zeros(mi.Float, dr.width(Le))
        ss_grad = mi.Spectrum(0.)
        st_grad = mi.Spectrum(0.)

        # Total distance traveled by the ray in the medium
        total_t = mi.Float(0.)
        # Number of scattering events
        n_scat = mi.Float(0)
        # Inside/outside of the target mesh status, assuming all rays start from outside
        inside_target = mi.Bool(False)

        active = mi.Bool(True)
        depth = mi.UInt32(0)
        active_medium = mi.Bool(False)

        while dr.hint(active, label=f"Backprojection ({mode.name})", exclude=[Le]):

            q = dr.minimum(0.99, dr.max(attenuation))
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr
            attenuation[perform_rr] *= dr.rcp(q)

            active &= dr.any(attenuation != 0)
            active_medium &= active

            # Find next intersection
            #TODO: needs_intersection mask
            si = scene.ray_intersect(ray, active=active)

            active &= si.is_valid()
            active_medium &= active

            hit_target = active & (si.shape == target_shape)

            weight = mi.Spectrum(1.)
            if dr.hint(has_scattering, mode='scalar'):
                mei = medium.sample_interaction(ray, sampler.next_1d(), 0, active_medium)
                reached_surface = active_medium & si.is_valid() & (si.t < mei.t)
                mei.t[reached_surface] = dr.inf
                tr, tr_pdf = medium.transmittance_eval_pdf(mei, si, active_medium)

                mei.t = dr.detach(mei.t)

                inv_pdf = dr.select(tr_pdf > 0.0, dr.detach(dr.rcp(tr_pdf)), 0.0)
                weight[active_medium] *= tr * inv_pdf
            else:
                reached_surface = mi.Bool(active_medium)
                mei = mi.MediumInteraction3f()
                mei.sigma_s, mei.sigma_n, mei.sigma_t = medium.get_scattering_coefficients(mei, active_medium)

            active_medium &= ~reached_surface

            g_em, g_ss, g_st = sensor.accumulate(ray, Le, inside_target, attenuation, total_t, n_scat, si.t, mei, sampler, active=active_medium | reached_surface, δL=δL, mode=mode)

            # Flip inside/outside flag if the target was hit
            inside_target = (~inside_target & hit_target) | (inside_target & ~hit_target) # /!\ This may cause some rays to leak out

            if dr.hint(is_backward, mode='scalar'):
                em_grad += g_em
                ss_grad += g_ss
                st_grad += g_st

            # BSDF Sampling

            active_surface = active & si.is_valid() & ~active_medium
            # Next bounce
            bsdf = si.bsdf(ray)
            ctx = mi.BSDFContext()
            s1 = sampler.next_1d(active_surface)
            s2 = sampler.next_2d(active_surface)

            if dr.hint(self.transmission_only, mode='scalar'):
                ctx.type_mask = mi.BSDFFlags.Transmission
                bs, bs_w = bsdf.sample(ctx, si, s1, s2, active_surface)
            else:
                # If this is the first intersection, we only sample a refraction to avoid having useless rays
                force_tr = active_surface & (depth == 0)
                bs, bs_w = bsdf.sample(ctx, si, s1, s2, active_surface & ~force_tr)
                ctx.type_mask = mi.BSDFFlags.Transmission
                bs[force_tr], bs_w[force_tr] = bsdf.sample(ctx, si, s1, s2, force_tr)

            weight[active_surface] *= bs_w

            ray[active_surface] = si.spawn_ray(si.to_world(bs.wo))

            if dr.hint(has_scattering, mode='scalar'):
                weight[active_medium] *= mei.sigma_s
                # Phase function sampling
                phase_ctx = mi.PhaseFunctionContext(sampler)
                phase = medium.phase_function()
                with dr.suspend_grad():
                    wo, phase_w = phase.sample(phase_ctx, mei,
                                                        sampler.next_1d(active_medium),
                                                        sampler.next_2d(active_medium),
                                                        active_medium)[:2]
                    weight[active_medium] *= phase_w
                    ray[active_medium] = mei.spawn_ray(wo)
                n_scat[active_medium] += 1
            else:
                weight[active & reached_surface] *= dr.exp(-mei.sigma_t * si.t)

            total_t[active_medium | reached_surface] += dr.select(reached_surface, si.t, mei.t)
            attenuation[active] *= dr.detach(weight)

            active_medium |= active_surface & (hit_target | (si.is_medium_transition() & (si.target_medium(ray.d) != None)))
            active &= (active_surface | active_medium)

            depth[active & ~hit_target] += 1
            active &= (depth < self.max_depth)

        if is_backward:
            if dr.grad_enabled(Le):
                dr.backward_from(Le * em_grad, dr.ADFlag.ClearInterior)
            if dr.grad_enabled(sigma_t):
                dr.backward_from(sigma_t * st_grad, dr.ADFlag.ClearInterior)
            if dr.grad_enabled(sigma_s):
                dr.backward_from(sigma_s * ss_grad)

        return sensor.film().develop() if is_primal else δL

mi.register_integrator("volumeintegrator", VolumeIntegrator)

