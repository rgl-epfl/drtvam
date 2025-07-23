import mitsuba as mi
import drjit as dr
import numpy as np
import glob
import os
from .motion import Motion, motions

def load_patterns(filepath):
    if os.path.isfile(filepath):
        if filepath.endswith(".npy"):
            patterns = np.load(filepath)
        elif filepath.endswith(".npz"):
            patterns = np.load(filepath)
            if len(patterns.files) != 1:
                raise ValueError(f"Expected a single array in the npz file, but got {len(patterns.files)} arrays.")
            patterns = patterns[patterns.files[0]]
        else:
            raise ValueError(f"Unsupported file format for patterns: {os.path.splitext(filepath)[1]}")

        if len(patterns.shape) != 3:
            raise ValueError(f"Patterns must be 3D, but got a tensor of shape {patterns.shape}.")
        return mi.TensorXf(patterns)

    filenames = glob.glob(os.path.join(filepath, "*.exr"))
    if len(filenames) == 0:
        raise ValueError("No patterns found in the specified directory. Please make sure the patterns are in EXR format.")

    N = len(filenames)
    for i, fn in enumerate(sorted(filenames)):
        img = mi.TensorXf(mi.Bitmap(fn))
        if i == 0:
            h, w, _ = img.shape
            imgs = dr.empty(mi.TensorXf, shape=(N,h,w))
            idx = dr.arange(mi.UInt32, h*w)
        elif img.shape[:2] != (h, w):
                raise ValueError(f"File '{fn}' has a different resolution ({img.shape[0]}x{img.shape[1]}) than the previous files ({h}x{w}). All patterns are expected to have the same resolution.")
        dr.scatter(imgs.array, img.array, idx + i*h*w)
    dr.eval(imgs)
    return imgs

class TVAMProjector(mi.Emitter):
    def __init__(self, props):
        super().__init__(props)

        self.m_sampler = props.get('sampler', mi.load_dict({'type': 'independent'}))

        if 'patterns' in props:
            if isinstance(props['patterns'], str):
                patterns = load_patterns(props['patterns'])
            elif isinstance(props['patterns'], mi.TensorXf):
                patterns = props['patterns']
            elif isinstance(props['patterns'], np.ndarray):
                patterns = mi.TensorXf(props['patterns'])
            else:
                raise ValueError("[self.__class__.__name__] patterns must be of type TensorXf")

            if len(patterns.shape) != 3:
                raise ValueError(f"[self.__class__.__name__] Patterns must be 3D, but got a tensor of shape {patterns.shape}.")

            n, h, w = patterns.shape
            self.n_patterns = n
            self.res = mi.ScalarVector2i(w, h)
            self.crop = self.res
            self.crop_offset = mi.ScalarVector2i(0, 0)

            if props.get('filter_nonzero', False):
                self.active_pixels = dr.compress(patterns.array > 0) + dr.opaque(mi.UInt32, 0)
                self.active_data = dr.gather(mi.Float, patterns.array, self.active_pixels)
            else:
                self.active_data = mi.Float(patterns.array)
                self.active_pixels = dr.arange(mi.UInt32, n*h*w)
        else:
            self.n_patterns = props.get('n_patterns', 1000)
            resx = props.get('resx', 256)
            resy = props.get('resy', 256)
            self.res = mi.ScalarVector2i(resx, resy)

            cropx = props.get('cropx', resx)
            cropy = props.get('cropy', resy)
            self.crop = mi.ScalarVector2i(cropx, cropy)
            if dr.any(self.crop > self.res):
                raise ValueError(f"[self.__class__.__name__] Crop resolution ({self.crop}) must be smaller than the base resolution ({self.res}).")

            crop_offset_x = props.get('crop_offset_x', 0)
            crop_offset_y = props.get('crop_offset_y', 0)
            self.crop_offset = mi.ScalarVector2i(crop_offset_x, crop_offset_y)
            if dr.any(self.crop_offset + self.crop > self.res):
                raise ValueError(f"[self.__class__name__] With the specified crop offset ({self.crop_offset}), the cropped region ({self.crop}) extends beyond the base resolution ({self.res}).")

            self.active_data = dr.zeros(mi.Float, self.n_patterns * dr.prod(self.crop))

            crop_idx = dr.arange(mi.UInt32, dr.prod(self.crop))
            crop_row_idx = crop_idx // self.crop.x
            crop_col_idx = crop_idx % self.crop.x
            crop_pixel_idx = (self.crop_offset.y + crop_row_idx) * self.res.x + crop_col_idx + self.crop_offset.x
            active_pixels = dr.tile(crop_pixel_idx, self.n_patterns)
            pattern_idx = dr.repeat(dr.arange(mi.UInt32, self.n_patterns), dr.prod(self.crop))
            self.active_pixels = pattern_idx * dr.prod(self.res) + active_pixels
            dr.eval(self.active_data, self.active_pixels)

        # Load projector motion
        if "motion" not in props:
            raise ValueError(f"[self.__class__.__name__] Missing field 'motion'.")
        if isinstance(props['motion'], Motion):
            self.motion = props['motion']
        elif isinstance(props['motion'], str):
            if props['motion'] not in motions.keys():
                raise ValueError(f"[self.__class__.__name__] Invalid motion type: {props['motion']}")
            self.motion = motions[props['motion']](props)
        else:
            raise ValueError(f"[self.__class__.__name__] motion must be either a dict or a Motion instance")

        self.m_flags = mi.EmitterFlags.DeltaDirection
        self.sample_to_camera = None

    def sampler(self):
        return self.m_sampler

    def active_size(self):
        return len(self.active_data)

    def size(self):
        return (self.n_patterns, self.res.y, self.res.x)

    def patterns(self):
        """Return the full patterns"""
        patterns = dr.zeros(mi.TensorXf, shape=(self.n_patterns, self.res.y, self.res.x))
        dr.scatter(patterns.array, self.active_data, self.active_pixels)
        return patterns

    def traverse(self, callback):
        callback.put_parameter("active_data", self.active_data, mi.ParamFlags.Differentiable)
        callback.put_parameter("active_pixels", self.active_pixels, mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        # check if active_mask_new has no True field where it was False before, i.e. we can't reactivate pixels after disabling them
        if len(self.active_data) != len(self.active_pixels):
            raise ValueError(f"[{self.__class__.__name__}] active_data and active_pixels must have the same length.")


    def get_ray(self, position_sample, aperture_sample):
        """
        Return the ray origin and direction in the reference frame (i.e. before the to_world transformation),
        as well as the reciprocal of the sampling pdf
        """
        raise NotImplementedError

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        """
        Given a time and position sample, return a ray and its associated weight.
        The position sample is in [0, 1]^2, and the time sample is in [0, 1].
        """
        #TODO: use active mask
        if self.sample_to_camera is None:
            raise ValueError(f"[{self.__class__.__name__}] This plugin should define a sample_to_camera matrix.")

        # Generate ray from the projection model
        origin, direction, inv_pdf = self.get_ray(position_sample, aperture_sample)

        # Get the current emitter world position and transform the ray accordingly
        to_world = self.motion.eval(time)
        ray = to_world @ mi.Ray3f(origin, direction)

        n_samples = dr.width(position_sample)
        return ray, inv_pdf / n_samples

class CollimatedProjector(TVAMProjector):
    def __init__(self, props):
        super().__init__(props)

        self.pixel_size = props['pixel_size']
        if isinstance(self.pixel_size, float): # Should this go to subclasses ?
            self.pixel_size = mi.Point2f(self.pixel_size)
        elif not isinstance(self.pixel_size, mi.Point2f):
            raise ValueError("[self.__class__.__name__] pixel_size must be a float or a Point2f")

        self.emitter_size = self.res * self.pixel_size
        aspect = self.res.x / self.res.y
        camera_to_sample = mi.orthographic_projection(self.res, self.res, mi.ScalarPoint2i(0), 1e-2, 1e4)
        scale = mi.Transform4f().scale(0.5 * mi.Point3f(self.emitter_size.x, self.emitter_size.y * aspect, 1.))

        self.sample_to_camera = scale @ camera_to_sample.inverse()

    def get_ray(self, position_sample, aperture_sample):
        origin = self.sample_to_camera @ mi.Point3f(position_sample.x, position_sample.y, 0.)
        direction = mi.Vector3f(0., 0., 1.)
        active_area = dr.prod(self.pixel_size) * len(self.active_data)
        return origin, direction, active_area

    def to_string(self):
        return ('CollimatedProjector[\n'
                f'    pattern count = {self.n_patterns},\n'
                f'    pattern resolution = {self.res},\n'
                f'    emitter_size = {self.emitter_size},\n'
                f'    sampler = {self.m_sampler},\n'
                ']')


class TelecentricProjector(TVAMProjector):
    def __init__(self, props):
        super().__init__(props)

        self.pixel_size = props['pixel_size']
        self.aperture_radius = props['aperture_radius']
        self.focus_distance = props['focus_distance']


        if isinstance(self.pixel_size, float): # Should this go to subclasses ?
            self.pixel_size = mi.Point2f(self.pixel_size)
        elif not isinstance(self.pixel_size, mi.Point2f):
            raise ValueError("[self.__class__.__name__] pixel_size must be a float or a Point2f")

        self.emitter_size = self.res * self.pixel_size
        aspect = self.res.x / self.res.y
        camera_to_sample = mi.orthographic_projection(self.res, self.res, mi.ScalarPoint2i(0), 1e-2, 1e4)
        scale = mi.Transform4f().scale(0.5 * mi.Point3f(self.emitter_size.x, self.emitter_size.y * aspect, 1.))

        self.sample_to_camera = scale @ camera_to_sample.inverse()

    def get_ray(self, position_sample, aperture_sample):
        origin = self.sample_to_camera @ mi.Point3f(position_sample.x, position_sample.y, 0.)

        # we sample from an aperture with the aperture radius
        scaled_aperture_sample = self.aperture_radius * mi.warp.square_to_uniform_disk_concentric(aperture_sample)
        # then we start the ray from within the aperture but we offset with the
        # respective pixel position
        scaled_aperture_sample_p = mi.Point3f(origin.x + scaled_aperture_sample.x, origin.y + scaled_aperture_sample.y, 0.0)

        # we need to specify the correct direction of the ray
        moved_scaled_aperture_sample = mi.Point3f(origin.x + scaled_aperture_sample.x, origin.y + scaled_aperture_sample.y, -self.focus_distance)
        direction = dr.normalize(-moved_scaled_aperture_sample + origin)

        active_area = dr.prod(self.pixel_size) * len(self.active_data)
        return scaled_aperture_sample_p, direction, active_area

    def to_string(self):
        return ('TelecentricProjector')



class LensProjector(TVAMProjector):
    def __init__(self, props):
        super().__init__(props)

        self.fov = props['fov'] #TODO: use another parameterization based on focal length and sensor size
        self.aperture_radius = props['aperture_radius']
        self.focus_distance = props['focus_distance']

        camera_to_sample = mi.perspective_projection(self.res, self.res, mi.ScalarPoint2i(0), self.fov, 1e-2, 1e4)
        self.sample_to_camera = camera_to_sample.inverse()

        pmin = self.sample_to_camera @ mi.Point3f(0., 0., 0.)
        pmax = self.sample_to_camera @ mi.Point3f(1., 1., 0.)
        image_rect = mi.BoundingBox2f(mi.Point2f(pmin.x, pmin.y) / pmin.z)
        image_rect.expand(mi.Point2f(pmax.x, pmax.y) / pmax.z)
        self.area = image_rect.volume()

    def get_ray(self, position_sample, aperture_sample):
        """
        Return the ray origin and direction in the reference frame (i.e. before the to_world transformation)
        """
        near_p = mi.Vector3f(self.sample_to_camera @ mi.Point3f(position_sample.x, position_sample.y, 0.))
        tmp = self.aperture_radius * mi.warp.square_to_uniform_disk_concentric(aperture_sample)

        origin = mi.Vector3f(tmp.x, tmp.y, 0.)
        focus_p = near_p * (self.focus_distance / near_p.z)
        direction = dr.normalize(focus_p - origin)

        return origin, direction, dr.pi * self.area

    def to_string(self):
        return ('LensProjector[\n'
                f'    fov = {self.fov},\n'
                f'    aperture radius = {self.aperture_radius},\n'
                f'    focus distance = {self.focus_distance},\n'
                f'    pattern count = {self.n_patterns},\n'
                f'    pattern resolution = {self.res},\n'
                f'    sampler = {self.m_sampler},\n'
                ']')


#TODO: unit tests
mi.register_emitter('collimated', CollimatedProjector)
mi.register_emitter('lens', LensProjector)
mi.register_emitter('telecentric', TelecentricProjector)
