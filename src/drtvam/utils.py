import mitsuba as mi
import drjit as dr
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def iou_loss(pred, target, threshold=0.9):
    obj_mask = target.array > 0.
    thresholded = pred.array > threshold
    return mi.Float(dr.count(thresholded & obj_mask)) / dr.count(thresholded | obj_mask)


def wasserstein_distance_volumes(target, vol):
    from scipy.stats import wasserstein_distance

    obj_mask = target.numpy().flatten() > 0.
    voxels_final = vol.numpy().flatten()

    p = voxels_final[obj_mask]
    q = voxels_final[~obj_mask]

    return wasserstein_distance(p, q)


def calculate_absorbed_dose(config, pattern_efficiency, target, vol, patterns):
    pixel_size = config["projector"]["pixel_size"]

    # projected intensity in drtvam units
    projected_intensity_drtvam_units = dr.sum(patterns) * pixel_size**2
    voxel_volume = (config["sensor"]["scalex"] / config["sensor"]["film"]["resx"] * \
                    config["sensor"]["scaley"] / config["sensor"]["film"]["resy"] * \
                    config["sensor"]["scalez"] / config["sensor"]["film"]["resz"])

    # absorbed intensity in target in drtvam units
    absorbed_intensity_target_drtvam_units = dr.sum(vol * target) * voxel_volume


    # amount of target voxels times voxel volume to get total volume of target
    total_voxel_volume_target = dr.sum(target) * voxel_volume * 1e-3**3

    # this ratio value needs to be between 0 and 1
    # it describes how much of the projected intensity is absorbed in the target volume
    # print("ratio", absorbed_intensity_target_drtvam_units / projected_intensity_drtvam_units)


    # I am not sure why but it seems like we need to multiply by 1e-3 to get
    # the units right? but it should be unitless already. But somehow it still
    # has a millimeter -> meter error in there.
    ratio = 1e-3 * absorbed_intensity_target_drtvam_units / projected_intensity_drtvam_units

    # needs need to be multiplied with total power of DMD times printing_time
    # to obtain J/m^3. Right now the units are 1/m^3
    absorbed_dose = pattern_efficiency * ratio / total_voxel_volume_target
    return absorbed_dose


def bhattacharyya_distance_coefficient(target, vol):
    # Create bins
    num_bins = 1000
    bins = np.linspace(0, 1.5, num_bins)

    obj_mask = target.numpy().flatten() > 0.
    voxels_final = vol.numpy().flatten()

    # Get values inside and outside object
    p = voxels_final[obj_mask]
    q = voxels_final[~obj_mask]

    # Create histograms
    p_b, _ = np.histogram(p, bins)
    q_b, _ = np.histogram(q, bins)

    # Normalize to probabilities
    p_b = p_b / np.sum(p_b)
    q_b = q_b / np.sum(q_b)

    # Calculate Bhattacharyya coefficient with epsilon
    bc = np.sum(np.sqrt(p_b * q_b))

    # Calculate distance with epsilon to avoid log(0)
    bd = -np.log(bc)

    return bd, bc



def reshape_grid(array):
    if len(array.shape) == 3:
        n, h, w = array.shape
        c = 1
    elif len(array.shape) == 4:
        n, h, w, c = array.shape
    else:
        raise ValueError(f"Invalid array shape: {array.shape}")

    rows = int(np.ceil(np.sqrt(n)))
    # Add padding to make the grid square
    array_new = np.zeros((rows**2, h, w, c))
    array_new[:n] = array

    return array_new.reshape((rows, rows, h, w, c)).swapaxes(1, 2).reshape((rows*h, rows*w, c))

def save_img(img, path):
    assert type(img) == mi.TensorXf
    if len(img.shape) == 2:
        bmp = mi.Bitmap(img[..., None])
    elif len(img.shape) == 3:
        bmp = mi.Bitmap(img)
    else:
        raise ValueError("Invalid image shape")
    bmp.write(path)

def save_vol(vol, path):
    if isinstance(vol, mi.TensorXf):
        vol = vol.numpy()
    elif not isinstance(vol, np.ndarray):
        raise ValueError(f"Invalid volume type: '{type(vol)}'")

    bmp = mi.Bitmap(mi.TensorXf(reshape_grid(vol)))
    bmp.write(path)

def save_histogram(vol, target, filename, efficiency, iou, thresholds, best_threshold, best_threshold_normalized):
    fig = plt.figure(figsize=(10, 5))
    obj_mask = target.numpy().flatten() > 0.
    voxels_final = vol.numpy().flatten()
    bins = np.linspace(0, 1, 500)
    plt.hist(voxels_final[obj_mask], bins=500, label="Object", alpha=0.55)
    plt.hist(voxels_final[~obj_mask], bins=500, label="Empty", alpha=0.55)

    plt.xlim([0, 1.2])
    plt.title("pattern energy efficiency = {:.4f}, IoU = {:.4f} at threshold = {:.3f}, normalized threshold = {:.3f}".\
              format(efficiency, iou, thresholds[best_threshold],
                     best_threshold_normalized))
    plt.yscale('log')
    plt.ylabel("# Voxels")
    plt.xlabel("Received dose")
    plt.legend()
    plt.savefig(filename)

def discretize(scene, sensor=0):
    """
    Given a scene containing a target shape, this function converts
    it to a binary occupancy grid, to be used as the reference for optimization.
    """
    if isinstance(sensor, int):
        sensor = scene.sensors()[sensor]

    target_shape = None
    for shape in scene.shapes():
        if dr.hint(shape.id() == 'target', mode='scalar'):
            target_shape = shape
    if target_shape is None:
        raise ValueError("No target shape found in the scene")

    bbox = target_shape.bbox()

    target_scene = mi.load_dict({
        'type': 'scene',
        'target': target_shape,
    })

    res = sensor.resolution()
    voxel_size = sensor.bbox.extents() / mi.Vector3f(res.x, res.y, res.z)

    xx = dr.arange(mi.Float, res.x)
    yy = dr.arange(mi.Float, res.y)
    zz = dr.arange(mi.Float, res.z)
    z_idx, y_idx, x_idx = dr.meshgrid(zz, yy, xx, indexing='ij')

    pos = sensor.bbox.min + (0.5 + mi.Point3f(x_idx, y_idx, z_idx)) * voxel_size

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, dr.width(pos))

    ray = mi.Ray3f(pos, mi.warp.square_to_uniform_sphere(sampler.next_2d()))
    active = dr.all((ray.o > bbox.min) & (ray.o < bbox.max))

    si = target_scene.ray_intersect(ray, active=active)

    inside = si.is_valid() & (si.shape == mi.ShapePtr(target_shape)) & (dr.dot(si.n, ray.d) > 0)

    voxels = dr.zeros(mi.TensorXf, shape=(res.z, res.y, res.x, 1))
    voxels.array[inside] = 1.0

    return voxels

def get_mesh_transform(filename, print_size, size=None):
    """
    Params
    ------

    filename: str
        Path to the mesh file

    print_size: mi.Point3f
        Size of the printable volume

    size: float
        Size of the object in the grid. If None, the object is scaled to fit the grid as tightly as possible.
    """
    ext = os.path.splitext(filename)[1][1:]
    if not ext in ['ply', 'obj']:
        raise ValueError(f"Unsupported extension: '{ext}', only PLY and OBJ meshes are supported.")

    shape = mi.load_dict({
            'type': ext,
            'filename': filename
    })
    params = mi.traverse(shape)
    bbox = shape.bbox()
    ext_max = dr.max(bbox.extents())
    if size is None:
        # Scale the object to fit the grid as tightly as possible
        # Align the largest dimension with the z-axis
        # This is because the printing volume is taller than it is wide because of refractions.
        axis_max = dr.select(bbox.extents() == ext_max, 1., 0.)

        if dr.any(axis_max != mi.ScalarVector3f(0, 0, 1)):
            rot_axis = dr.cross(axis_max, mi.ScalarVector3f(0, 0, 1))
            rot_mat = mi.ScalarTransform4f().rotate(rot_axis, 90.)
        else:
            rot_mat = mi.ScalarTransform4f(1.)

        # Find the enclosing circle of the object in the XY plane
        v = params['vertex_positions'].numpy().reshape((-1, 3))
        mask = (axis_max.numpy() != np.ones(3))
        v_2d = v[:, mask]
        import miniball
        mb = miniball.Miniball(v_2d)
        c = np.zeros(3)
        c[mask] = mb.center()
        c[~mask] = 0.5 * (bbox.min + bbox.max).numpy()[~mask]
        r = np.sqrt(mb.squared_radius())

        tr = mi.ScalarTransform4f().translate(-c)
        scale = mi.Point3f()
        scale.z = print_size.z / ext_max # Scale factor so the height fits on the DMD
        scale.xy = 0.5 * print_size.xy / r # Scale factor so the enclosing circle fits in the grid
        scale = mi.ScalarTransform4f().scale(dr.min(scale)[0])
        to_world = scale @ rot_mat @ tr
    else:
        if dr.any(size > print_size):
            raise ValueError("The size of the object is larger than the printable volume.")
        c = 0.5 * (bbox.min + bbox.max)
        to_world = mi.ScalarTransform4f().scale(size / ext_max) @ mi.ScalarTransform4f().translate(-c)

    return to_world

