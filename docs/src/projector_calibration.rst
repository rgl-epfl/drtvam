.. _projector_calibration:

Projector Calibration
=====================
This tutorial describes how you can calibrate a ``telecentric`` and ``lens`` (perspective) projector.



Trace single projector pixels
-----------------------------
Dr.TVAM has the option to trace single rays instead of doing an optimization.
The following example shows how to trace rays for a single projector pixel.

.. code-block:: json

    {
        "vial": {
            "type": "cylindrical",
            "r_int": 16.363125,
            "r_ext": 17.354374999999999,
            "ior": 1.00,
            "medium": {
                "ior": 1.00,
                "phase": {
                    "type": "rayleigh"
                },
                "extinction": 0.1,
                "albedo": 0.0
            }
        },
        "projector": {
            "type": "telecentric",
            "n_patterns": 1,
            "resx": 740,
            "resy": 700,
            "aperture_radius": 4.0,
            "pixel_size": 20.54e-3,
            "motion": "circular",
            "distance": 150,
            "focus_distance": 150.0
        },
        "sensor": {
            "type": "dda",
            "scalex": 15.2,
            "scaley": 15.2,
            "scalez": 15.2,
            "film": {
                "type": "vfilm",
                "resx": 1000,
                "resy": 1000,
                "resz": 1
            }
        },
        "target": {
            "filename": "cylinder.ply",
            "size": 1000
        },
        "psf_analysis": [
            {
                "x": 370,
                "y": 350,
                "index_pattern": 0,
                "intensity": 1
            },
            {
                "x": 0,
                "y": 350,
                "index_pattern": 0,
                "intensity": 1
            },
            {
                "x": 739,
                "y": 350,
                "index_pattern": 0,
                "intensity": 1
            }
        ],
        "spp_ref": 10000
    }


Notable we introduce a ``psf_analysis`` section in the JSON file.
This section contains a list of rays to be traced. Each ray is defined by its ``x`` and ``y`` pixel coordinates, the ``index_pattern`` (which pattern to use), and the ``intensity`` of the ray.
In this case we would turn on the most left pixel, the middle pixel, and the most right pixel of the projector pattern.
If ``drtvam config.json`` is run, it will only trace the rays defined in the ``psf_analysis`` section.
The output intensity traces are written in ``final.exr`` and ``final.npy``. The pixel size of the output is defined by the sensor,
hence the output will be 1000x1000 pixels with a pixel size of 0.0152 mm.
Since the refractive index of the vial is 1.0, the rays will not be refracted and will travel in a straight line, as expected in air.
The rays are attenuated by the extinction coefficient of the medium, which is set to 0.1.

The target is irrelevant for this example, but it is required to run the simulation.

We shoot a total of 10000 rays per pixel, as defined by the ``spp_ref`` parameter. It is possible to change this value to increase or decrease the number of rays per pixel. It makes the results more accurate.
Note, in an optimization, increasing the ``spp`` parameters to such high values, will result in very long optimizations. 
So values around 100 (``spp=100``, ``spp_ref=100`` and ``spp_grad=100``) are more realistic and sufficient for most applications.


Calibration a real ``lens`` projector
-------------------------------------
This example shows how to calibrate a :ref:`real lens projector <lens_projector>` setup.
