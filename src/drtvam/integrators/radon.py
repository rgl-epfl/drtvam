from __future__ import annotations as __annotations__ # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr
from .common import TVAMIntegrator

class RadonIntegrator(TVAMIntegrator):
    def to_string(self):
        return ('RadonIntegrator[\n'
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

        projector = scene.emitters()[0]

        with dr.suspend_grad():
            sampler, spp = self.prepare(projector, seed, spp)
            ray, _, weight = self.sample_rays(scene, projector, sampler)

            L = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                active=mi.Bool(True)
            ) * weight

            imgs = dr.zeros(mi.TensorXf, projector.size())
            idx = dr.repeat(projector.active_pixels, spp)
            dr.scatter_reduce(dr.ReduceOp.Add, imgs.array, L, idx)

        return imgs

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float]]:

        medium, target_shape = self.parse_scene(scene)
        if target_shape is None:
            raise ValueError("No target shape found in the scene")
        sigma_s, _, sigma_t = medium.get_scattering_coefficients(mi.MediumInteraction3f())

        active = mi.Bool(True)
        active_medium = mi.Bool(False)
        inside_target = mi.Bool(False)
        throughput = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        t = mi.Float(0.0)
        d = mi.UInt32(0)

        while active:
            si = scene.ray_intersect(ray)

            active &= si.is_valid()

            # Here we make the assumption that the medium is purely absorptive
            contrib = throughput * dr.exp(-sigma_t*t) * (1 - dr.exp(-sigma_t * si.t))
            hit_target = active & (si.shape == target_shape)
            L[inside_target & active_medium] += contrib
            t[active & active_medium] += si.t

            inside_target = (~inside_target & hit_target) | (inside_target & ~hit_target) # /!\ This may cause some rays to leak out

            bsdf = si.bsdf(ray)
            ctx = mi.BSDFContext()
            s1 = sampler.next_1d(active)
            s2 = sampler.next_2d(active)

            if dr.hint(self.transmission_only, mode='scalar'):
                ctx.type_mask = mi.BSDFFlags.Transmission
                bs, bs_w = bsdf.sample(ctx, si, s1, s2, active)
            else:
                # If this is the first intersection, we only sample a refraction to avoid having useless rays
                force_tr = active & (depth == 0)
                bs, bs_w = bsdf.sample(ctx, si, s1, s2, active & ~force_tr)
                ctx.type_mask = mi.BSDFFlags.Transmission
                bs[force_tr], bs_w[force_tr] = bsdf.sample(ctx, si, s1, s2, force_tr)

            throughput[active] *= bs_w

            ray[active] = si.spawn_ray(si.to_world(bs.wo))

            d[active & ~hit_target] += 1
            active &= (d < self.max_depth)

            active_medium = (active_medium & ~si.is_medium_transition()) | (si.is_medium_transition() & (si.target_medium(ray.d) != None))

        return L

mi.register_integrator("radon", RadonIntegrator)

