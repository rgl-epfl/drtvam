import drjit as dr
import mitsuba as mi
import numpy as np
import os

from drtvam.convolution import make_drjit_conv

def assemble_physical_forward_model(config):
    if "physical_model" in config and config["physical_model"]["type"] == "inhibitor":
        inhibitor_0 = dr.ones(mi.TensorXf,
                              shape=(config["sensor"]["film"]["resz"],
                                     config["sensor"]["film"]["resy"],
                                     config["sensor"]["film"]["resx"],1))
        if "inhibitor_init" in config["physical_model"]:
            print("Loading inhibitor initial state from",
                  config["physical_model"]["inhibitor_init"])
            inhibitor_0 *= mi.TensorXf(np.load(config["physical_model"]["inhibitor_init"]))

        def physical_fwd(light_dose):
            q_inhibitor = dr.minimum(light_dose, inhibitor_0)
            inhibitor = inhibitor_0 - q_inhibitor
            polymerization = light_dose - q_inhibitor
            return polymerization, inhibitor

    elif "physical_model" in config and config["physical_model"]["type"] == "inhibitor_diffusion":
        inhibitor_0 = dr.ones(mi.TensorXf,
                              shape=(config["sensor"]["film"]["resz"],
                                     config["sensor"]["film"]["resy"],
                                     config["sensor"]["film"]["resx"],1))
        if "inhibitor_init" in config["physical_model"]:
            print("Loading inhibitor initial state from",
                  config["physical_model"]["inhibitor_init"])
            inhibitor_0 *= mi.TensorXf(np.load(config["physical_model"]["inhibitor_init"]))

        print("Using diffusion model for inhibition.")
        diffusion_D = config['physical_model']['diffusion_coefficient']
        diffusion_time = config['print_time']
        diffusion_number_rotations = config['physical_model']['number_time_steps']

        spacingz = config['sensor']['scalez'] / config['sensor']['film']['resz']
        spacingx = config['sensor']['scalex'] / config['sensor']['film']['resx']
        spacingy = config['sensor']['scaley'] / config['sensor']['film']['resy']


        delta_t = diffusion_time / diffusion_number_rotations

        # rough estimate of how many pixels the diffusion kernel should cover, based on the diffusion distance
        radiusz = int(np.sqrt(6 * diffusion_D * delta_t) / spacingz * 3)
        radiusx = int(np.sqrt(6 * diffusion_D * delta_t) / spacingx * 3)
        radiusy = int(np.sqrt(6 * diffusion_D * delta_t) / spacingy * 3)

        conv = make_drjit_conv(spacingz, spacingy, spacingx, diffusion_D, delta_t,
            radiusz, radiusy, radiusx)

        def physical_fwd(light_dose):
            # see https://drjit.readthedocs.io/en/stable/autodiff.html#differentiating-loops
            def loop(light_dose, diffusion_number_rotations):
                inhibitor = inhibitor_0
                polymerization = 0 * inhibitor
                i = 0

                light_dose_loop = light_dose / diffusion_number_rotations
                while dr.hint(i < diffusion_number_rotations, max_iterations=-1):
                    q_inhibitor = dr.minimum(light_dose_loop, inhibitor)
                    inhibitor = inhibitor - q_inhibitor
                    radicals = light_dose_loop - q_inhibitor
                    polymerization += radicals
                    # Diffusion step
                    inhibitor = conv(inhibitor)
                    i += 1

                return polymerization, inhibitor
            return loop(light_dose, diffusion_number_rotations)
    else:
        # default cause, just return light dose
        def physical_fwd(light_dose):
            return (light_dose, )

    return physical_fwd



# legacy torch code to perform convolution
#       x = (torch.arange(config['sensor']['film']['resx'], device='cuda')) * spacingx - config['sensor']['scalex'] / 2
        # y = (torch.arange(config['sensor']['film']['resy'], device='cuda')) * spacingy - config['sensor']['scaley'] / 2
        # z = (torch.arange(config['sensor']['film']['resz'], device='cuda')) * spacingz - config['sensor']['scalez'] / 2
        # Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

        # delta_t = diffusion_time / diffusion_number_rotations
        # r = torch.sqrt(X**2 + Y**2 + Z**2)

        # diffusion_kernel = torch.exp(-r**2 / (4 * diffusion_D * delta_t))
        # diffusion_kernel /= torch.sum(diffusion_kernel)

        # diffusion_kernel = torch.fft.ifftshift(diffusion_kernel)
        # diffusion_kernel = diffusion_kernel[:, :, :, None]

        # diffusion_kernel_drjit = dr.cuda.TensorXf(diffusion_kernel)
        # np.save(os.path.join(config["output"], "diffusion_kernel.npy"),
        #         np.fft.fftshift(diffusion_kernel_drjit.numpy()))
        # conv = lambda x: fft_convolve_3d(x, diffusion_kernel_drjit)
