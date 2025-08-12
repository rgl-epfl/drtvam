import drtvam
import mitsuba as mi
import drjit as dr
import numpy as np
import os
from tqdm import trange
import json
import argparse

from drtvam.geometry import geometries
from drtvam.utils import save_img, save_vol, save_histogram, discretize
from drtvam.loss import losses
from drtvam.lbfgs import LinearLBFGS

def load_scene(config):
    for key in ['target', 'vial', 'projector', 'sensor']:
        if key not in config:
            raise ValueError(f"Missing field '{key}' in the configuration file.")

    # Load vial geometry
    if 'type' not in config['vial']:
        raise ValueError("The vial geometry must have a 'type' field.")
    if config['vial']['type'] not in geometries.keys():
        raise ValueError(f"Unknown vial geometry: '{config['vial']['type']}'")

    vial = geometries[config['vial']['type']](config['vial'])

    if 'filename' not in config['target']:
        raise ValueError("Missing field 'filename' for the target shape.")

    # Target mesh transform
    mesh_type = os.path.splitext(config['target']['filename'])[1][1:]
    bbox = mi.load_dict({
        'type': mesh_type,
        'filename': config['target']['filename']
    }).bbox()

    c = 0.5 * (bbox.min + bbox.max)
    size = config['target'].get('size', 1.)
    center_pos_x = config['target'].get('box_center_x', 0.)
    center_pos_y = config['target'].get('box_center_y', 0.)
    center_pos_z = config['target'].get('box_center_z', 0.)

    center_pos = mi.ScalarPoint3f(center_pos_x, center_pos_y, center_pos_z)
    # Scale and center the target object
    # first translate to the center of the bounding box
    # then scale to the size of the bounding box
    # then translate to user specified position (if there is one)
    target_to_world = mi.ScalarTransform4f().translate(center_pos) @ \
      mi.ScalarTransform4f().scale(size / dr.max(bbox.extents())) @ mi.ScalarTransform4f().translate(-c)

    def get_sensor_transform(sensor_dict):
        sensor_scalex = sensor_dict.pop('scalex', 1.)
        sensor_scaley = sensor_dict.pop('scaley', 1.)
        sensor_scalez = sensor_dict.pop('scalez', 1.)
        return mi.ScalarTransform4f().scale(mi.ScalarPoint3f(sensor_scalex, sensor_scaley, sensor_scalez))

    sensor_to_world = get_sensor_transform(config['sensor'])

    # Create Mitsuba scene
    scene_dict = {
        'type': 'scene',
        'projector': config['projector'],
        'sensor': config['sensor'] | {'to_world': sensor_to_world},
        'target': {
            'type': mesh_type,
            'filename': config['target']['filename'],
            'to_world': target_to_world,
            'bsdf': {
                'type': 'null'
            }
        },
    } | vial.to_dict()

    if 'final_sensor' in config.keys():
        final_sensor_to_world = get_sensor_transform(config['final_sensor'])
        scene_dict['final_sensor'] = config['final_sensor'] | {'to_world': final_sensor_to_world}

    return scene_dict

def optimize(config, patterns_fwd=None):
    """

    Optimize the patterns for the TVAM.
    Args:
        config (dict): Configuration dictionary containing the scene and optimization parameters.
        patterns_fwd (np.ndarray, optional): if provided, the actual optimization is skipped
    """
    scene_dict = load_scene(config)
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    output = config['output']

    # Rendering parameters
    spp = config.get('spp', 4)
    spp_ref = config.get('spp_ref', 16)
    spp_grad = config.get('spp_grad', spp)
    max_depth = config.get('max_depth', 6)
    rr_depth = config.get('rr_depth', 6) # i.e. disabled by default
    time = config.get('time', 1.) # Print duration in seconds
    progressive = config.get('progressive', False)
    transmission_only = config.get('transmission_only', True)
    regular_sampling = config.get('regular_sampling', False)
    sensor = None
    final_sensor = None
    for s in scene.sensors():
        if s.id() == 'sensor':
            sensor = s
        elif s.id() == 'final_sensor':
            final_sensor = s

    if final_sensor is None:
        final_sensor = sensor
    if final_sensor.film().surface_aware:
        raise ValueError("The final sensor is used to generate visualizations and metrics of the final simulated print. Therefore, it must not be surface-aware. If you are using the surface-aware discretization for optimization, please specify another sensor called 'final_sensor' in the configuration file.")

    surface_aware = sensor.film().surface_aware
    filter_radon = config.get('filter_radon', False) # Disable DMD pixels where the Radon transform is zero

    integrator = mi.load_dict({
        'type': 'volumeintegrator',
        'max_depth': 3 if progressive else max_depth,
        'rr_depth': rr_depth,
        'print_time': time,
        'transmission_only': transmission_only,
        'regular_sampling': regular_sampling
    })

    # Computing reference
    if surface_aware:
        target = sensor.compute_volume(scene)
        save_vol(target[..., 0, None], os.path.join(output, "target_in.exr"))
        save_vol(target[..., 1, None], os.path.join(output, "target_out.exr"))
    else:
        target = discretize(scene, sensor=sensor)
        save_vol(target, os.path.join(output, "target.exr"))

    np.save(os.path.join(output, "target.npy"), target.numpy())

    patterns_key = 'projector.active_data'

    if filter_radon and patterns_fwd is None:
        # Deactivate pixels where the Radon transform is zero
        radon_integrator = mi.load_dict({
            'type': 'radon',
            'max_depth': 5,
        })
        radon = mi.render(scene, integrator=radon_integrator, spp=config.get('spp_filter_radon', 4))

        active_pixels = dr.compress(radon.array > 0.) + dr.opaque(mi.UInt32, 0) # Hack to get the result of compress to only use its actual size
        dr.eval(active_pixels)

        if len(active_pixels) == 0:
            raise ValueError("No active pixels found in the Radon transform.")

        params['projector.active_pixels'] = active_pixels
        params[patterns_key] = dr.zeros(mi.Float, dr.width(active_pixels))
        params.update()

        del radon, radon_integrator
        dr.flush_malloc_cache()
        dr.sync_thread()


    if 'filter_corner' in config and patterns_fwd is None:
        corner_integrator = mi.load_dict({
            'type': 'corner',
            'regular_sampling': True,
        } | config['filter_corner'])
        corner = mi.render(scene, integrator=corner_integrator, spp=1)

        active_pixels = dr.compress(corner.array > 0.) + dr.opaque(mi.UInt32, 0) # Hack to get the result of compress to only use its actual size
        dr.eval(active_pixels)

        if len(active_pixels) == 0:
            raise ValueError("No active pixels found in the Radon transform.")

        params['projector.active_pixels'] = active_pixels
        params[patterns_key] = dr.zeros(mi.Float, dr.width(active_pixels))
        params.update()

        del corner, corner_integrator
        dr.flush_malloc_cache()
        dr.sync_thread()


    # If not using the surface-aware discretization, we don't need the target shape anymore, so we just move it far away
    if not surface_aware:
        params['target.vertex_positions'] += 1e5
        params.update()

    if "loss" not in config.keys():
        print("No loss function specified. Using thresholded loss.")
        config['loss'] = {'type': 'threshold'}

    loss_type = config['loss'].pop('type')
    if loss_type not in losses.keys():
        raise ValueError(f"Unknown loss type: '{loss_type}'. Available losses are: {list(losses.keys())}")

    loss_fn = losses[loss_type](config['loss'])

    if 'optimizer' not in config.keys():
        print("No optimizer specified. Using linear L-BFGS.")
        config['optimizer'] = {'type': 'lbfgs'}

    optim_type = config['optimizer'].pop('type')
    if optim_type == 'adam':
        opt = mi.ad.Adam(**config['optimizer'])
    elif optim_type == 'sgd':
        opt = mi.ad.SGD(**config['optimizer'])
    else:
        def render_fn(vars):
            params[patterns_key] = vars[patterns_key]
            params.update()
            vol = mi.render(scene, params, integrator=integrator, sensor=sensor, spp=spp, spp_grad=spp_grad, seed=i)
            return vol

        def loss_fn2(y, patterns):
            return loss_fn(y, target, patterns)

        opt = LinearLBFGS(loss_fn=loss_fn2, render_fn=render_fn)

    # Pass patterns to optimizer
    opt[patterns_key] = params[patterns_key]
    n_steps = config.get('n_steps', 40)

    loss_hist = np.zeros(n_steps)
    timing_hist = np.zeros((n_steps, 2))

    integrator_final = mi.load_dict({
        'type': 'volumeintegrator',
        'max_depth': config.get('max_depth_ref', 16),
        'rr_depth': config.get('rr_depth_ref', 8),
        'transmission_only': transmission_only,
        'regular_sampling': regular_sampling,
        'print_time': time
    })

    if patterns_fwd is not None:
        print("Using provided patterns for forward mode.")
        params['projector.active_data'] = patterns_fwd.flatten()
        params.update()

    elif "psf_analysis" in config:
        print("\nPSF analysis enabled.")
        print("Exporting ray tracing...")
        # we simply loop over the entries specified in the json
        number_rays_psf = len(config["psf_analysis"])
        print("Number of traced pixels:", number_rays_psf)
        params['projector.active_data'] = dr.ones(mi.UInt32, number_rays_psf)
        params['projector.active_pixels'] = dr.zeros(mi.UInt32, number_rays_psf)

        xres = config["projector"]["resx"]
        yres = config["projector"]["resy"]
        for (i, entry) in enumerate(config["psf_analysis"]):
            assert entry["x"] < xres, "Invalid entry in psf_analysis: x out of bounds. Please check the configuration file."
            assert entry["y"] < yres, "Invalid entry in psf_analysis: y out of bounds. Please check the configuration file."
            assert entry["index_pattern"] < config["projector"]["n_patterns"], \
                "Invalid entry in psf_analysis: index_pattern out of bounds. Please check the configuration file."

            params['projector.active_pixels'][i] = \
                xres * yres * entry["index_pattern"] + xres * entry["y"] + entry["x"]
            params['projector.active_data'][i] *= entry["intensity"]

        params.update()
        print("Rendering final state...")
        vol_final = mi.render(scene, params, spp=spp_ref, integrator=integrator_final, sensor=final_sensor)

        np.save(os.path.join(output, "final.npy"), vol_final.numpy())
        save_vol(vol_final, os.path.join(output, "final.exr"))

        np.save(os.path.join(output, "loss.npy"), loss_hist)
        np.save(os.path.join(output, "timing.npy"), timing_hist)

        imgs_final = scene.emitters()[0].patterns()
        dr.eval(imgs_final)

        print("Saving images...")
        for i in trange(imgs_final.shape[0]):
            save_img(imgs_final[i], os.path.join(output, "patterns", f"{i:04d}.exr"))
        np.savez_compressed(os.path.join(output, "patterns.npz"), patterns=imgs_final.numpy())

        return vol_final
    else:
        print("Optimizing patterns...")
        for i in trange(n_steps):
            if progressive and i == 5:
                integrator.max_depth = max_depth

            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                params.update(opt)

                vol = mi.render(scene, params, integrator=integrator, sensor=sensor, spp=spp, spp_grad=spp_grad, seed=i)
                dr.schedule(vol)

                mi.Log(mi.LogLevel.Debug, "[drtvam] Calling loss from optimize loop")
                loss = loss_fn(vol, target, params['projector.active_data'])
                dr.eval(loss)

                # numpy conversion is necessary to store the loss value
                # apparently in just loss.numpy() is deprecated since (Deprecated NumPy 1.25.)
                loss_hist[i] = loss[0].numpy()

                # Primal timing
                timing_hist[i, 0] = sum([h['execution_time'] for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT])

                dr.backward(loss)

                if dr.all(loss == 0):
                    print("Converged")
                    break

                if optim_type == 'lbfgs':
                    opt.step(vol, loss)
                else:
                    opt.step()

                # Clamp patterns
                opt[patterns_key] = dr.maximum(dr.detach(opt[patterns_key]), 0)

                # Adjoint timing
                timing_hist[i, 1] = sum([h['execution_time'] for h in dr.kernel_history() if h['type'] == dr.KernelType.JIT])
        params.update(opt)


    print("Rendering final state...")
    vol_final = mi.render(scene, params, spp=spp_ref, integrator=integrator_final, sensor=final_sensor)

    np.save(os.path.join(output, "final.npy"), vol_final.numpy())
    save_vol(vol_final, os.path.join(output, "final.exr"))

    np.save(os.path.join(output, "loss.npy"), loss_hist)
    np.save(os.path.join(output, "timing.npy"), timing_hist)

    imgs_final = scene.emitters()[0].patterns()
    dr.eval(imgs_final)

    print("Saving images...")
    for i in trange(imgs_final.shape[0]):
        save_img(imgs_final[i], os.path.join(output, "patterns", f"{i:04d}.exr"))
    np.savez_compressed(os.path.join(output, "patterns.npz"), patterns=imgs_final.numpy())

    # save also the compressed version normalized to [0, 255]
    # Step 1: Normalize the array to [0, 1]
    array = imgs_final.numpy()
    array_max = np.max(array)
    normalized_array = array / array_max
    # Step 2: Scale to [0, 255]
    scaled_array = normalized_array * 255
    # Step 3: Convert to np.uint8
    final_array = scaled_array.astype(np.uint8)
    np.savez_compressed(os.path.join(output, "patterns_normalized_uint8.npz"), patterns=final_array)

    # save a high resolution in case of surface aware since the resolution
    # might be low of target.exr/npy
    if surface_aware:
        target = discretize(scene, sensor=final_sensor)
        np.save(os.path.join(output, "target_binary.npy"), target.numpy())
        save_vol(target, os.path.join(output, "target_binary.exr"))

    efficiency = np.sum(normalized_array / normalized_array.size)
    print("Pattern efficiency {:.4f}".format(efficiency))

    save_histogram(vol_final, target, os.path.join(output, "histogram.png"),
                   efficiency, array_max)

    return vol_final




class OverrideAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.overrides = {}

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            key, value = values.split('=')
        except ValueError:
            raise ValueError("Invalid parameter override. Use the format '-D key=value'")

        # Try to convert the value to a number if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass # Keep the value as a string

        self.overrides[key] = value
        setattr(namespace, self.dest, self.overrides)

def main():
    parser = argparse.ArgumentParser("Optimize patterns for TVAM.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("-D", dest="overrides", metavar="key=value", action=OverrideAction, help="Override/Add a parameter in the configuration dictionary. Nested keys are separated by dots.")
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "llvm"], help="Select the backend for the optimization.")
    parser.add_argument("--forward_mode", action="store_true", help="Just project the patterns without optimization.\
                        Patterns need to be specified by --patterns (a .npz file).")
    parser.add_argument("--patterns", type=str, help="Path to the patterns file (a .npz file). This is only used in forward mode.")

    args = parser.parse_args()

    mi.set_variant(f"{args.backend}_ad_mono")

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Apply overrides
    if args.overrides is not None:
        for key, value in args.overrides.items():
            key = key.split('.')
            tmp = config
            for k in key[:-1]:
                tmp = tmp[k]
            tmp[key[-1]] = value

    # Add the directory of the configuration file to the file resolver for relative paths
    mi.Thread.thread().file_resolver().append(os.path.dirname(os.path.abspath(args.config)))

    if 'output' not in config:
        config['output'] = os.path.dirname(os.path.abspath(args.config))

    # Save the configuration file in the output directory
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)
    with open(os.path.join(config['output'], "opt_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    if args.forward_mode:
        # Forward mode: just project the patterns
        if 'patterns' not in args:
            raise ValueError("In forward mode, you must specify the patterns file.")
        patterns = np.load(args.patterns)['patterns']
        optimize(config, patterns_fwd=patterns)
    else:
        # Run the optimization
        optimize(config)


if __name__ == "__main__":
    main()

