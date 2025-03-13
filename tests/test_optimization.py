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



@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_square_hole_occlusion_optimization(variant):
    mi.set_variant(variant)

    fname = 'tests/files/box_hole_occlusion.json'
    # Load the configuration file
    with open(fname, 'r') as f:
        config = json.load(f)

    config['output'] = os.path.dirname(os.path.abspath(fname))

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
                                   'tests/files/box_hole_square_weighted_loss.json',\
                                   'tests/files/box_hole_index_matched.json'\
                                   ])
@pytest.mark.parametrize("variant", ["cuda_ad_mono", "llvm_ad_mono"])
def test_square_hole_optimization(fname, variant):
    mi.set_variant(variant)

    # Load the configuration file
    with open(fname, 'r') as f:
        config = json.load(f)

    config['output'] = os.path.dirname(os.path.abspath(fname))

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


    # scattering works less well
    if fname == 'tests/files/box_hole_scattering.json':
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

