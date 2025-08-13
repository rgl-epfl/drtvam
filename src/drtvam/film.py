import mitsuba as mi
import drjit as dr

class VolumetricFilm(mi.Film):
    def __init__(self, props):
        super().__init__(props)

        #TODO: have proper reconstruction filters
        resz = props.get('resz', 256)
        resy = props.get('resx', 256)
        resx = props.get('resy', 256)

        # Spatial resolution of the film
        self.res = mi.ScalarVector3i(resx, resy, resz)

        # Use surface-aware discretization ?
        self.surface_aware = props.get('surface_aware', False)
        if self.surface_aware:
            self.data = dr.zeros(mi.TensorXf, (resz, resy, resx, 2))
        else:
            self.data = dr.zeros(mi.TensorXf, (resz, resy, resx, 1))

    def to_string(self):
        return ('VolumetricFilm[\n'
                f'    resolution = {self.data.shape},\n'
                ']')

    def clear(self):
        self.data = dr.zeros(mi.TensorXf, self.data.shape)

    def traverse(self, callback):
        callback.put("data", self.data, mi.ParamFlags.Differentiable)

    def resolution(self):
        return self.res

    def develop(self):
        return self.data

    def write(self, values, idx, active):
        dr.scatter_reduce(dr.ReduceOp.Add, self.data.array, values, idx, active)

mi.register_film('vfilm', VolumetricFilm)

