import mitsuba as mi
import drjit as dr
import pytest
import sys
import os
import json

# sys.path.append(os.path.join(os.path.dirname(__file__), '../drtvam'))
import drtvam
from drtvam.loss import *
from drtvam.optimize import *
from drtvam.utils import discretize
import matplotlib.pyplot as plt

# None = default optimizer (LinearLBFGS)
OPTIMIZERS = [None,
              {'type': 'lbfgs', 'wolfe': False},
              {'type': 'lbfgs', 'wolfe': True}]




@pytest.mark.parametrize("fname", ['tests/files/double_cylindrical.json'])
@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_double_cylindrical(fname, variant, optimizer):
    mi.set_variant(variant)

    # Load the configuration file
    with open(fname, 'r') as f:
        config = json.load(f)

    config['output'] = os.path.dirname(os.path.abspath(fname))
    if optimizer is not None:
        config['optimizer'] = dict(optimizer)

    # Save the configuration file in the output directory
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)
    with open(os.path.join(config['output'], "opt_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    vol_final = optimize(config)

    reference = 1.0 * (np.load("tests/files/target_hollow_gear.npy"))
    almost_equal = np.isclose(reference, 1.0 * (vol_final.numpy() > 0.6))
    percentage_correct = np.mean(almost_equal) * 100
    assert percentage_correct > 99.4



@pytest.mark.parametrize("fname", ['tests/files/box_hole_occlusion.json',\
                                   "tests/files/box_hole_occlusion_filter_radon.json"])
@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_square_hole_occlusion_optimization(fname, variant, optimizer):
    mi.set_variant(variant)

    # Load the configuration file
    with open(fname, 'r') as f:
        config = json.load(f)

    config['output'] = os.path.dirname(os.path.abspath(fname))
    if optimizer is not None:
        config['optimizer'] = dict(optimizer)

    # Save the configuration file in the output directory
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)
    with open(os.path.join(config['output'], "opt_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    vol_final = optimize(config)

    # voxelized reference
    reference = np.zeros((50, 100, 100))
    reference[5:45, 10:90, 10:90] = 1


    occlusion = np.zeros((50, 100, 100))
    occlusion[15:35, 40:60, 30:70] = 1

    reference = reference - occlusion

    size = (100, 100)
    radius = 20
    center_x, center_y = 10 + radius, 50  # Center coordinates
    # Create a meshgrid for the coordinates
    y, x = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

    # Use the circle equation to create the circular mask
    mask = (x - center_y) ** 2 + (y - center_x) ** 2 < (radius + 0.5) ** 2
    # Initialize the array and apply the mask
    array = np.zeros((50, 100, 100), dtype=int)
    array[5:45, mask] = 1

    reference[:, :, :] = (reference - array) > 0

    almost_equal = np.isclose(reference, vol_final[:, :, :, 0] >
                              (config['loss']["tl"] + config['loss']["tu"]) / 2)
    percentage_correct = np.mean(almost_equal) * 100

    # plt.show()
    # plt.figure()
    # plt.imshow(reference[25, :, :])
    # plt.figure()
    # plt.imshow(almost_equal[25, :, :])
    # # plt.imshow(vol_final.numpy()[25, :, :, 0])
    # plt.figure()
    # plt.show()
    print(percentage_correct)
    assert percentage_correct > 97.0




@pytest.mark.parametrize("fname", ['tests/files/box_hole_scattering.json',\
                                   'tests/files/box_hole_cylindrical.json',\
                                   'tests/files/box_hole_square.json',\
                                   'tests/files/box_hole_square_different_thresholds.json',\
                                   #'tests/files/box_hole_square_weighted_loss.json',\
                                   'tests/files/box_hole_index_matched.json',\
                                   'tests/files/box_hole_custom_cuvette.json'\
                                   ])
@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_square_hole_optimization(fname, variant, optimizer):
    mi.set_variant(variant)

    # Load the configuration file
    with open(fname, 'r') as f:
        config = json.load(f)

    config['output'] = os.path.dirname(os.path.abspath(fname))
    if optimizer is not None:
        config['optimizer'] = dict(optimizer)

    # Save the configuration file in the output directory
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)
    with open(os.path.join(config['output'], "opt_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    vol_final = optimize(config)

    # voxelized reference
    reference = np.zeros((50, 100, 100))
    reference[5:45, 10:90, 10:90] = 1
    size = (100, 100)
    radius = 20
    center_x, center_y = 10 + radius, 50  # Center coordinates
    # Create a meshgrid for the coordinates
    y, x = np.meshgrid(np.arange(size[1]), np.arange(size[0]))

    # Use the circle equation to create the circular mask
    mask = (x - center_y) ** 2 + (y - center_x) ** 2 < (radius + 0.5) ** 2
    # Initialize the array and apply the mask
    array = np.zeros((50, 100, 100), dtype=int)
    array[5:45, mask] = 1

    reference[:, :, :] = (reference - array)

    almost_equal = np.isclose(reference, vol_final[:, :, :, 0] >
                              (config['loss']["tl"] + config['loss']["tu"]) / 2)
    percentage_correct = np.mean(almost_equal) * 100


    # scattering and custom cuvette works less well
    if fname == 'tests/files/box_hole_scattering.json' or fname == 'tests/files/box_hole_custom_cuvette.json':
        assert percentage_correct > 99.0
    else:
        assert percentage_correct > 99.4

    # plt.figure()
    # plt.figure()
    # plt.imshow(almost_equal[:, 50, :])
    # plt.show()
    # plt.figure()
    # plt.imshow(reference[:, 50, :])
    # plt.figure()
    # plt.imshow(vol_final.numpy()[:, 50, :, 0])
    # plt.show()

@pytest.mark.parametrize("max_pattern_value", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_max_pattern_value(max_pattern_value, variant, optimizer, tmp_path):
    mi.set_variant(variant)

    with open('tests/files/box_hole_square.json', 'r') as f:
        config = json.load(f)

    config['output'] = str(tmp_path)
    if optimizer is not None:
        config['optimizer'] = dict(optimizer)
    config['projector']['max_pattern_value'] = max_pattern_value
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)

    optimize(config)

    patterns = np.load(os.path.join(config['output'], "patterns.npz"))['patterns']

    assert np.min(patterns) >= 0.0
    assert np.max(patterns) <= max_pattern_value


# These setups can reach negative or zero curvature, but LBFGS optimizers expect it to be positive to progress.
# Hence, in this case we stop the optimization early to prevent patterns becoming NaN.
@pytest.mark.parametrize("fname, overrides", [
    ('tests/files/early_stopping.json', {}),
    ('tests/files/box_hole_square.json',
     {'optimizer': {'type': 'lbfgs', 'wolfe': True, 'init_alpha': 1.0}, 'n_steps': 300}),
])
@pytest.mark.parametrize("variant", ["cuda_ad_mono"])
def test_early_stopping(fname, overrides, variant, tmp_path):
    mi.set_variant(variant)

    with open(fname, 'r') as f:
        config = json.load(f)
    config.update(overrides)
    config['output'] = str(tmp_path)
    os.makedirs(os.path.join(config['output'], "patterns"), exist_ok=True)

    vol_final = optimize(config)

    patterns = np.load(os.path.join(config['output'], "patterns.npz"))['patterns']
    assert not np.isnan(patterns).any()

    target = np.load(os.path.join(config['output'], "target.npy"))
    threshold = (config['loss']['tl'] + config['loss']['tu']) / 2
    assert np.mean((target > 0.5) == (vol_final.numpy() > threshold)) * 100 > 99.4
