.. _real_world_examples:

Real World Examples
===================

In this section we explain in detail how we use this toolbox to print a real boat. 
This is only one example, for more details and options see the other :doc:`tutorials <tutorials>`. and the :doc:`plugin documentation <plugin_doc>`.


Resin Preparation
-----------------
For the resin we use the commercial *Sartomer Arkema resin* which mainly consists of *Dipentaerythritol pentaaycrlate*.
As photo initiator we use TPO.
With a refractometer we measure the refractive index :math:`n_\text{resin} = 1.4849`.
We pour the resin into a cup. The photoinitiator is mixed into IPA. This is shaken until the TPO is dissolved.
The IPA with the TPO is mixed into the resin. It is mixed in a Kurabo Planetary Mixer for some minutes. 
In total, we mix roughly :math:`30\mathrm{mg}` of TPO into :math:`40\mathrm{mL}` of the resin.
With a spectrometer, we determine the absorbance at our printing wavelength :math:`405\mathrm{nm}` to be :math:`A=0.2347/1\mathrm{cm}`. That means, :math:`\mu = ln(10)\cdot A / 10 \approx 0.05404\mathrm{mm}^{-1}`-
Technically there is also absorption of the resin itself which does not contribute to the absorption but we determined it to be :math:`A=1.92\mathrm{m^{-1}}`.
So we neglect this effect and assume all absorbed light is contributing to the polymerization.

.. image:: resources/container.jpeg
  :width: 400


Glass Vial
----------
As glass vials we use simple cylindrical glass vial which are not quite optimized for optical applications.
With a measurement calliper we determine the outer radius to be :math:`R_\text{outer} = (8.3\pm0.01)\mathrm{mm}` and the inner radius
:math:`R_\text{outer} = (7.6\pm 0.01)\mathrm{mm}`. The refractive index is roughly :math:`n_\text{vial}=1.58`.

.. image:: resources/vial.jpeg
  :width: 400


DMD Characterization
--------------------
We have a camera system which images the printing path at the center of the vial.
After our 4f printing optical system, the DMD pixel pitch is :math:`13.79\mathrm{\mu m}`.


Selecting a Target
------------------
In this case we optimize the  3D Benchy (https://3dbenchy.com/) as a printing target.



Specifying Optimization Parameters
----------------------------------
.. _real_world_examples_spp:

We specify the optimization parameters with a JSON file. Below is an example of a JSON file which we use for the optimization:
.. code-block:: 

    {
        # defines vial parameters
         "vial": {
             # cylindrical vial and no index matching bath
             "type": "cylindrical",
             "r_int": 7.6,
             "r_ext": 8.3,
             # refractive index of the vial
             "ior": 1.58,
             # describes the medium of the resin
             "medium": {
                 "ior": 1.4849,
                 # phase function in case of scattering
                 "phase": {
                     "type": "rayleigh"
                 },
                 # we are using mm as units, so this is mm^-1
                 "extinction": 0.054, 
                 # albedo indicates no scattering
                 "albedo": 0.0
             }
         },
         # printing illimination
         "projector": {
             # suitable for a laser
             "type": "collimated",
             # amount of different angular patterns from [0°, 360°) 
             "n_patterns": 300,
             # resolution of the projector
             "resx": 400,
             "resy": 400,
             # pixel pitch in mm
             "pixel_size": 0.0137,
             "motion": "circular",
             # distance is irrelevant for a collimated beam
             "distance": 20
         },
         # sensor effectively corresponds to the discretization of the target 
         "sensor": {
             "type": "dda",
             # size in mm of the region where we track the absorption
             "scalex": 5,
             "scaley": 5,
             "scalez": 5,
             "film": {
                 "type": "vfilm",
                 "resx": 256,
                 "resy": 256,
                 "resz": 256,
                # can be set to true and will reduce discretization artefacts.
                # it can speed up the optimization a lot!
                 "surface_aware": false
             },
         },
         # target to print
         "target": {
             "filename": "/home/felix/Documents/data/sparse_tests_benchy/benchy.ply",
             # it takes a bounding box around the target and scales the largest
             # dimension to the given size in mm
             "size": 5.0
         },
         "loss": {
             "type": "threshold",
             "tl": 0.88,
             "tu": 0.95,
             # no sparsity enforced
             "weight_sparsity": 0.0,
             "M": 4
         },
         # filter radon can reduce computational time since we ignore "black" pixels
         "filter_radon": true,
         "progressive": true,
         "n_steps": 30,
         # how many rays are shot per pixel -> important for scattering. 
         "spp": 4,
         "spp_ref": 4,
         "spp_grad": 4
     }

Here the valid JSON without comments:

.. code-block:: json

    {
        "vial": {
            "type": "cylindrical",
            "r_int": 7.6,
            "r_ext": 8.3,
            "ior": 1.58,
            "medium": {
                "ior": 1.4849,
                "phase": {
                    "type": "rayleigh"
                },
                "extinction": 0.054,
                "albedo": 0.0
            }
        },
        "projector": {
            "type": "collimated",
            "n_patterns": 300,
            "resx": 400,
            "resy": 400,
            "pixel_size": 0.0137,
            "motion": "circular",
            "distance": 20
        },
        "sensor": {
            "type": "dda",
            "scalex": 5,
            "scaley": 5,
            "scalez": 5,
            "film": {
                "type": "vfilm",
                "resx": 256,
                "resy": 256,
                "resz": 256,
                "surface_aware": false 
            }
        },
        "target": {
            "filename": "lol/lel/foo/benchy.ply",
            "size": 5.0
        },
        "loss": {
            "type": "threshold",
            "tl": 0.88,
            "tu": 0.95,
            "weight_sparsity": 0.0,
            "M": 4
        },
        "progressive": true,
        "n_steps": 30,
        "spp": 4,
        "spp_ref": 4,
        "spp_grad": 4
    }


Lauching the Optimization
-------------------------
Open your terminal and laucnh the optimization with the following command. Of course, adapt the path

.. code-block:: bash

    $ drtvam lol/lel/foo/config.json
        No optimizer specified. Using L-BFGS.
        Optimizing patterns...
        100%|█████████████████████████████████| 30/30 [04:22<00:00,  8.75s/it]
        Rendering final state...
        Saving images...
        100%|█████████████████████████████████| 300/300 [00:01<00:00, 273.60it/s]
        Pattern efficiency 0.0359
        Finding threshold for best IoU ...
        best IoU: 1.0000
        best threshold: 0.913514

On a RTX 3060 this code runs for roughly 5min. GPUs with ray tracing cores and more VRAM allow for faster and larger simulations.

Analysing Results
-----------------

One standard check after the optimization is the histogram

.. image:: resources/histogram.png
  :width: 600

The orange part is the histogram of the intensity values of the void regions. The blue part is the histogram of the intensity values of the printed regions.
Both are well separated, which is a good sign for a successful optimization. If one hits the intensity spot of 0.914, the intersection over union (IoU) is 1.0.
The energy efficieny of the patterns is :math:`3.6\%`.

By default we export `.exr` images and `.npy` files. To view the `.exr` files we recommend using `tev <https://github.com/Tom94/tev>`_.

Also the file `final.exr` is insightful, as it displays the energy distribution in the vial for all slices.
Note, this file is potentially big and requires lots of VRAM or RAM to open.

.. image:: resources/final_exr.png
  :width: 600

The final patterns look like this (reduced size `.gif`):

.. image:: resources/patterns.gif
   :alt: StreamPlayer
   :align: center


Everything is optimal in this case. If the sparsity of the patterns is too high, try out to play with `weight_sparsity` in the JSON file.
