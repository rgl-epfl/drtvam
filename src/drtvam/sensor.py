import mitsuba as mi
import drjit as dr
from .film import VolumetricFilm

class VolumetricSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)
        if not isinstance(self.m_film, VolumetricFilm):
            raise ValueError("Tried to load a VolumetricSensor with a non-volumetric film. The film must be of type VolumetricFilm.")

        self.to_world = props.get('to_world', mi.ScalarTransform4f())

        # TODO: automatic scaling somewhere else
        corner_min = self.to_world @ mi.ScalarPoint3f(-0.5)
        corner_max = self.to_world @ mi.ScalarPoint3f(0.5)
        self.bbox = mi.BoundingBox3f(corner_min, corner_max)

        # Size of the voxel *along each dimension*
        self.voxel_size = self.bbox.extents() / mi.Vector3f(self.m_film.resolution())

        # Voxel volume
        self.volumes = None

    def traverse(self, callback):
        callback.put("to_world", self.to_world, mi.ParamFlags.NonDifferentiable)
        callback.put("film", self.m_film, mi.ParamFlags.Differentiable)

    def resolution(self):
        return self.m_film.resolution()

    def accumulate(self,
                   ray,
                   emitted,
                   inside_target,
                   attenuation,
                   t_prev,
                   n_scat,
                   maxt,
                   mei,
                   sampler,
                   active=True,
                   δL=None,
                   mode=dr.ADMode.Primal
                   ):
        raise NotImplementedError()

    @dr.syntax
    def compute_volume(self, scene: mi.Scene, sample_count=2**14):

        if dr.hint(not self.m_film.surface_aware, mode='scalar'):
            return dr.prod(self.voxel_size)

        # Surface-aware discretization
        if dr.hint(self.volumes is not None, mode='scalar'):
            return self.volumes

        target_shape = None
        for shape in scene.shapes():
            if dr.hint(shape.id() == 'target', mode='scalar'):
                target_shape = shape
        if target_shape is None:
            raise ValueError("No target shape found in the scene")

        target_scene = mi.load_dict({
            'type': 'scene',
            'target': target_shape,
        })

        bbox = target_shape.bbox()
        # First channel is "outside" and second channel is "inside"
        res = self.m_film.resolution()
        self.volumes = dr.ones(mi.TensorXf, shape=self.m_film.data.shape)

        xx = dr.arange(mi.Float, res.x)
        yy = dr.arange(mi.Float, res.y)
        zz = dr.arange(mi.Float, res.z)
        z_idx, y_idx, x_idx = dr.meshgrid(zz, yy, xx, indexing='ij')
        voxel = mi.Point3f(x_idx, y_idx, z_idx)

        voxel_vol = dr.prod(self.voxel_size)
        count_in = dr.zeros(mi.UInt32, dr.prod(res))
        count_out = dr.zeros(mi.UInt32, dr.prod(res))
        d = mi.UInt32(0)
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(0, dr.prod(res))

        active = mi.Bool(True)
        while active:
            offset = mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
            pos = self.bbox.min + self.voxel_size * (voxel + offset)
            ray = mi.Ray3f(pos, mi.warp.square_to_uniform_sphere(sampler.next_2d()))

            # If the ray origin is outside of its bounding box, we already know it's outside the mesh
            in_mesh_bbox = dr.all((ray.o > bbox.min) & (ray.o < bbox.max))
            si = target_scene.ray_intersect(ray, active=active & in_mesh_bbox)

            is_inside = si.is_valid() & (si.shape == mi.ShapePtr(target_shape)) & (dr.dot(ray.d, si.n) > 0)

            count_in[active & is_inside] += 1
            count_out[active & ~is_inside] += 1

            d[active] += 1
            active &= (d < sample_count)

        idx = dr.arange(mi.UInt32, dr.prod(res))
        dr.scatter(self.volumes.array, count_in * voxel_vol / sample_count, 2*idx)
        dr.scatter(self.volumes.array, count_out * voxel_vol / sample_count, 2*idx+1)

        dr.eval(self.volumes)
        return self.volumes

class DeltaVolumetricSensor(VolumetricSensor):
    def __init__(self, props):
        super().__init__(props)

    def to_string(self):
        return ('DeltaVolumetricSensor[\n'
                f'    to_world = {self.to_world},\n'
                ']')

    @dr.syntax
    def accumulate(self,
                   ray,
                   emitted,
                   inside_target,
                   attenuation,
                   t_prev,
                   n_scat,
                   maxt,
                   mei,
                   sampler,
                   active=True,
                   δL=None,
                   mode=dr.ADMode.Primal
                   ):

        g_em = dr.zeros(mi.Float, dr.width(emitted))
        g_st = dr.zeros(mi.Spectrum, dr.width(emitted))
        g_ss = dr.zeros(mi.Spectrum, dr.width(emitted))

        active = mi.Bool(active & mei.is_valid())
        res = self.m_film.resolution()
        pos = ray(mei.t)
        current_voxel = mi.Vector3i(dr.floor((pos - self.bbox.min) / self.voxel_size))
        is_inside_grid = dr.all(current_voxel >= 0) & dr.all(current_voxel < res)
        current_voxel_flat = current_voxel.x + current_voxel.y * res.x + current_voxel.z * res.x * res.y

        if dr.hint(self.m_film.surface_aware, mode='scalar'):
            idx = dr.select(inside_target, 2*current_voxel_flat, 2*current_voxel_flat+1)
        else:
            idx = current_voxel_flat

        em = dr.detach(emitted)
        ss = dr.detach(mei.sigma_s)
        st = dr.detach(mei.sigma_t)
        dr.set_grad_enabled(em, dr.grad_enabled(emitted))
        dr.set_grad_enabled(ss, dr.grad_enabled(mei.sigma_s))
        dr.set_grad_enabled(st, dr.grad_enabled(mei.sigma_t))

        if dr.hint(mode == dr.ADMode.Forward, mode='scalar'):
            dr.set_grad(em, emitted.grad)
            if dr.grad_enabled(mei.sigma_t):
                dr.forward_to(mei.sigma_t, mei.sigma_s)
            elif dr.grad_enabled(mei.sigma_s):
                dr.forward_to(mei.sigma_s)
            dr.set_grad(ss, mei.sigma_s.grad)
            dr.set_grad(st, mei.sigma_t.grad)

        sa = st - ss
        tr = dr.exp(-st * mei.t)
        tr_pdf = tr * st
        inv_pdf = dr.select(tr_pdf != 0, dr.detach(dr.rcp(tr_pdf)), 0.) # This is wrong if we don't sample free-flight distances proportional to transmittance
        throughput = dr.detach(attenuation * dr.exp(st * t_prev)) * dr.exp(-st * t_prev)
        throughput *= dr.select(ss != 0, ss**n_scat / dr.detach(ss**n_scat), 1.)

        contrib = dr.select(active, throughput * sa * tr * inv_pdf * em, 0.)
        if dr.hint(mode == dr.ADMode.Primal, mode='scalar'):
            self.m_film.write(contrib, idx, active & is_inside_grid)
        else:
            if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                # Reverse-mode AD
                grad = dr.gather(mi.Spectrum, δL.array, idx, active & is_inside_grid)
                dr.backward_from(contrib * grad)
                g_em = em.grad
                g_ss = ss.grad
                g_st = st.grad
            else:
                # Forward-mode AD
                dr.scatter_reduce(dr.ReduceOp.Add, δL.array, dr.forward_to(contrib), idx, active & is_inside_grid)

        return g_em, g_ss, g_st

class RatioVolumetricSensor(VolumetricSensor):
    def __init__(self, props):
        super().__init__(props)

        self.majorant = props['majorant']

    def to_string(self):
        return ('RatioVolumetricSensor[\n'
                f'    to_world = {self.to_world},\n'
                f'    majorant = {self.majorant},\n'
                ']')

    @dr.syntax
    def accumulate(self,
                   ray,
                   emitted,
                   inside_target,
                   attenuation,
                   t_prev,
                   n_scat,
                   maxt,
                   mei,
                   sampler,
                   active=True,
                   δL=None,
                   mode=dr.ADMode.Primal
                   ):
        is_primal = (mode == dr.ADMode.Primal)
        is_forward = (mode == dr.ADMode.Forward)
        active = mi.Bool(active)

        n_interactions = mi.Float(0)

        # Undo transmittance estimate from previous interactions, since we will recompute it
        throughput = dr.detach(attenuation * dr.exp(mei.sigma_t * t_prev))
        throughput *= dr.select(mei.sigma_s != 0, dr.rcp(dr.detach(mei.sigma_s) ** n_scat), 1.)

        if is_forward:
            if dr.grad_enabled(mei.sigma_t):
                dr.forward_to(mei.sigma_t, mei.sigma_s)
            elif dr.grad_enabled(mei.sigma_s):
                dr.forward_to(mei.sigma_s)

        g_em = dr.zeros(mi.Float, dr.width(emitted))
        g_st = dr.zeros(mi.Spectrum, dr.width(emitted))
        g_ss = dr.zeros(mi.Spectrum, dr.width(emitted))

        grid_res = self.m_film.resolution()
        t = mi.Float(0.)

        while dr.hint(active, label="Ratio tracking estimator", exclude=[emitted, mei]):
            sampled_t = - dr.log(1-sampler.next_1d(active)) / self.majorant

            t[active] += sampled_t

            active &= t < maxt

            p = ray(t)

            current_voxel = mi.Vector3i(dr.floor((p - self.bbox.min) / self.voxel_size))

            is_inside_grid = dr.all(current_voxel >= 0) & dr.all(current_voxel < grid_res)

            current_voxel_flat = current_voxel.x + current_voxel.y * grid_res.x + current_voxel.z * grid_res.x * grid_res.y
            if self.m_film.surface_aware:
                idx = dr.select(inside_target, 2*current_voxel_flat, 2*current_voxel_flat+1)
            else:
                idx = current_voxel_flat

            em = dr.detach(emitted)
            ss = dr.detach(mei.sigma_s)
            st = dr.detach(mei.sigma_t)
            if dr.hint(not is_primal, mode='scalar'):
                dr.set_grad_enabled(em, dr.grad_enabled(emitted))
                dr.set_grad_enabled(st, dr.grad_enabled(mei.sigma_t))
                dr.set_grad_enabled(ss, dr.grad_enabled(mei.sigma_s))
                if dr.hint(is_forward, mode='scalar'):
                    dr.set_grad(em, emitted.grad)
                    dr.set_grad(st, mei.sigma_t.grad)
                    dr.set_grad(ss, mei.sigma_s.grad)

            sa = st - ss
            weight = throughput * dr.exp(-st * t_prev)
            weight *= dr.select(ss != 0, ss ** n_scat, 1.)
            contrib = dr.select(active & is_inside_grid, weight * em * sa / st * (1 - st / self.majorant) ** n_interactions * st / self.majorant, 0.)

            if dr.hint(is_primal, mode='scalar'):
                self.m_film.write(contrib, idx, active & is_inside_grid)
            else:
                if dr.hint(is_forward, mode='scalar'):
                    # Forward-mode AD
                    dr.scatter_reduce(dr.ReduceOp.Add, δL.array, dr.forward_to(contrib), idx, active & is_inside_grid)
                else:
                    # Reverse-mode AD
                    grad = dr.gather(mi.Spectrum, δL.array, idx, active & is_inside_grid)
                    dr.backward_from(contrib * grad)
                    g_em += em.grad
                    g_ss += ss.grad
                    g_st += st.grad

            n_interactions[active] += 1

        return g_em, g_ss, g_st

class DDAVolumetricSensor(VolumetricSensor):
    def __init__(self, props):
        super().__init__(props)

    def to_string(self):
        return ('DDAVolumetricSensor[\n'
                f'    to_world = {self.to_world},\n'
                ']')

    @dr.syntax
    def accumulate(self,
                   ray,
                   emitted,
                   inside_target,
                   attenuation,
                   t_prev,
                   n_scat,
                   maxt,
                   mei,
                   sampler,
                   active=True,
                   δL=None,
                   mode=dr.ADMode.Primal
                   ):
        # TODO: use dr.dda here
        active = mi.Bool(active)

        is_primal = (mode == dr.ADMode.Primal)
        is_forward = (mode == dr.ADMode.Forward)

        # Find intersection with the bounding box of the volume grid
        t_bmin = (self.bbox.min - ray.o) / ray.d
        t_bmax = (self.bbox.max - ray.o) / ray.d

        mint_box = dr.maximum(dr.max(dr.minimum(t_bmin, t_bmax)), 0.)
        maxt_box = dr.min(dr.maximum(t_bmin, t_bmax))

        t_start = dr.maximum(mint_box, 0.)
        t_end = dr.minimum(maxt_box, maxt)

        active &= dr.isfinite(t_start) & dr.isfinite(t_end) & (t_start < t_end)

        grid_start = ray(t_start)
        grid_end = ray(t_end)

        grid_res = self.m_film.resolution()
        step_dir = dr.select(ray.d > 0, 1, -1)

        start_voxel = dr.clip(mi.Vector3i((grid_start - self.bbox.min) / self.voxel_size), 0, grid_res - 1)
        end_voxel = dr.clip(mi.Vector3i((grid_end - self.bbox.min) / self.voxel_size), 0, grid_res - 1)

        # Find the next voxel boundary
        next_voxel_pos = self.bbox.min + (start_voxel + step_dir) * self.voxel_size

        # If the ray has negative directions, we need to hit the left of the current
        # voxel, not the next one /!\ This is slightly different than the reference
        # algorithm, where they jump to the next cell instead of modifying the
        # boundary, but it should be equivalent.
        next_voxel_pos += dr.select(ray.d < 0, self.voxel_size, 0)

        is_valid_dir = dr.abs(ray.d) > 1e-8
        dtmax = dr.select(is_valid_dir, (next_voxel_pos - grid_start) / ray.d, dr.inf)
        dtmax[dtmax < 0] = dr.inf
        tstep = dr.select(is_valid_dir, self.voxel_size / ray.d * step_dir, dr.inf)

        current_voxel = mi.Vector3i(start_voxel)
        t = mi.Float(t_start)
        x = mi.Point3f(grid_start)
        remaining_dist = mi.Float(t_end - t_start)

        # Undo transmittance estimate from previous interactions, since we will recompute it
        throughput = dr.detach(attenuation * dr.exp(mei.sigma_t * t_prev))
        throughput *= dr.select(mei.sigma_s != 0, dr.rcp(dr.detach(mei.sigma_s) ** n_scat), 1.)

        if dr.hint(is_forward, mode='scalar'):
            if dr.hint(dr.grad_enabled(mei.sigma_t), mode='scalar'):
                dr.forward_to(mei.sigma_t, mei.sigma_s)
            elif dr.hint(dr.grad_enabled(mei.sigma_s), mode='scalar'):
                dr.forward_to(mei.sigma_s)

        g_em = dr.zeros(mi.Float, dr.width(emitted))
        g_ss = mi.Spectrum(0.)
        g_st = mi.Spectrum(0.)

        em = dr.detach(emitted)

        while dr.hint(active, label="DDA", exclude=[emitted, mei]):

            dt = dr.minimum(dr.min(dtmax), remaining_dist)
            remaining_dist[active] -= dt

            st = dr.detach(mei.sigma_t)
            ss = dr.detach(mei.sigma_s)
            em = dr.detach(emitted)
            if dr.hint(not is_primal, mode='scalar'):
                dr.set_grad_enabled(em, dr.grad_enabled(emitted))
                dr.set_grad_enabled(ss, dr.grad_enabled(mei.sigma_s))
                dr.set_grad_enabled(st, dr.grad_enabled(mei.sigma_t))
                if dr.hint(is_forward, mode='scalar'):
                    dr.set_grad(em, emitted.grad)
                    dr.set_grad(ss, mei.sigma_s.grad)
                    dr.set_grad(st, mei.sigma_t.grad)

            sa = st - ss
            weight = throughput * dr.exp(-st * t_prev)
            weight *= dr.select(ss != 0, ss ** n_scat, 1.)
            # Compute analytic absorption along the ray within the current voxel
            contrib = dr.select(active, weight * sa / st * em * dr.exp(-st*t) * (1 - dr.exp(-st * dr.maximum(dt, 0.))), 0.)
            current_voxel_flat = current_voxel.x + current_voxel.y * grid_res.x + current_voxel.z * grid_res.x * grid_res.y
            if self.m_film.surface_aware:
                idx = dr.select(inside_target, 2*current_voxel_flat, 2*current_voxel_flat+1)
            else:
                idx = current_voxel_flat

            if dr.hint(is_primal, mode='scalar'):
                self.m_film.write(contrib, idx, active)
            else:
                if dr.hint(is_forward, mode='scalar'):
                    # Forward-mode AD
                    dr.scatter_reduce(dr.ReduceOp.Add, δL.array, dr.forward_to(contrib), idx, active)
                else:
                    # Reverse-mode AD
                    grad = dr.gather(mi.Spectrum, δL.array, idx, active)
                    dr.backward_from(contrib * grad)
                    g_em += em.grad
                    g_ss += ss.grad
                    g_st += st.grad

                em = dr.detach(em)

            active &= dr.any(end_voxel != current_voxel) & (remaining_dist > 1e-6)

            voxel_update = mi.Vector3i()
            mask = dtmax == dt
            dtmax = dr.select(mask, tstep, dtmax - dt)
            voxel_update = dr.select(mask, step_dir, 0)

            current_voxel[active] += voxel_update

            active &= dr.all(current_voxel >= 0) & dr.all(current_voxel < grid_res)

            t[active] += dt

        return g_em, g_ss, g_st

mi.register_sensor('delta', DeltaVolumetricSensor)
mi.register_sensor('ratio', RatioVolumetricSensor)
mi.register_sensor('dda', DDAVolumetricSensor)

