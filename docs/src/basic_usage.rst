.. _basic_usage:

Basic Usage
===========

After installing Dr.TVAM, a ``drtvam`` command-line tool is available to run an
optimization from a configuration file. You can download an :download:`example
configuration file <resources/example_config.json>` with its associated
:download:`target shape <resources/benchy.ply>` to get started.

This configuration file is a JSON file that specifies the optical setup, and the
optimization parameters, whose entries we outline below. After defining it,
running the optimization is as simple as:

.. code-block:: bash

    drtvam path/to/config.json

This will run the optimization and save the results in the directory specified
in the configuration file, or in the directory containing the configuration file
if no output directory is specified.

Any entry in the JSON configuration file can be overridden by using the
``-Dkey=value`` optional argument. For example, if we were to override the
number of optimization steps and the output directory, we could run:

.. code-block:: bash

    drtvam path/to/config.json -Dn_steps=100 -Doutput=path/to/output

This script will try to run on the GPU if possible, and fall back to the CPU if not. If you want to speficy a particular backend, you can use the ``--backend <backend>`` command-line option, where ``<backend>`` can either be ``llvm`` (CPU) or ``cuda`` (GPU). For example, to run on the CPU, you can use: 

.. code-block:: bash

    drtvam path/to/config.json --backend llvm

Configuration syntax
--------------------
The JSON configuration must contain a number of dictionary entries, specifying
various aspects of the optimization:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description
    
    * - ``vial``
      - Dictionary
      - Contains parameters relative to the resin and its container. See the
        :ref:`vial` section for more details.
    
    * - ``projector``
      - Dictionary
      - Contains parameters relative to the projection system. See the
        :ref:`projector` section for more details.
    
    * - ``sensor``
      - Dictionary
      - Contains parameters relative to the sensor, which defines where the
        absporption measurements are taken. See the :ref:`sensor` section for
        more details. In particular, one can specify the resolution of the
        sensor in the nested ``film`` dictionary, and the scale of the sensor in
        each dimension in the ``scalex``, ``scaley`` and ``scalez`` entries. The
        sensor is a 3D grid centered at the origin, originally spanning from
        -0.5 to 0.5 in each dimension. The scale parameters allow to stretch the
        sensor in each dimension.
    
    * - ``target`` 
      - Dictionary
      - Contains the path to the target shape (e.g. ``"filename": "boat.ply"``), 
        in PLY or OBJ format, as well as
        its size (e.g. ``"size": 4.0``) along its largest axis. 
        The shape will be centered  at the
        origin and scaled such that its size in the largest dimension is equal
        to the specified size. The target shape is used to compute the target
        absorption map. If this dict contains the entries ``"box_center_x": 2``, 
        ``"box_center_y": 3``, and ``"box_center_z": 4``, the target shape box 
        will be centered at the coordinates ``(2, 3, 4)`` after the target
        has been scaled to ``size``.
        If ``scale_and_center`` is set to ``false``, the target shape will not be
        scaled or centered, by default it is set to ``true``.

Other entries are optional:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description
    
    *  - ``loss``
       - Dictionary
       - Which loss should be used, with the syntax ``'type': 'loss_name'``,
         along with key-value pairs for each additional parameter for the chosen
         loss function. See the :ref:`loss` section for the available losses. If
         not specified, the default loss is the thresholded loss from Wechsler
         et al. `[2024]
         <https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-14705&id=548744>`_.

    *  - ``optimizer``
       - Dictionary
       - Contains the choice of optimizer. Defaults to L-BFGS.

    *  - ``n_steps``
       - ``int``
       - The number of optimization steps. Defaults to 40.

    *  - ``spp``
       - ``int``
       - How many light paths are drawn per projector pixel in the forward
         evaluation of the model. Defaults to 4.

    *  - ``spp_grad``
       - ``int``
       - How many light paths are drawn per projector pixel in the
         backpropagation of the model. Defaults to ``spp``.

    *  - ``spp_ref``
       - ``int``
       - How many light paths are drawn per projector pixel when evaluating the
         final results. Defaults to 16.

    *  - ``max_depth``
       - ``int``
       - Maximum number of scattering events (surface or medium) computed before
         ending a path. For purely absorptive media, a value of 3 is often
         sufficient. Defaults to 6.

    *  - ``max_depth_ref``
       - ``int``
       - Same as ``max_depth`` but for final rendering

    *  - ``rr_depth``
       - ``int``
       - Light paths can be ended stochastically using "Russian Roulette" after
         this depth. Defaults to 6, i.e. it is disabled by default.

    *  - ``rr_depth_ref``
       - ``int``
       - Same as ``rr_depth`` but for final rendering

    *  - ``time``
       - ``float``
       - Print duration, in seconds. This defines the total exposure time.
         Defaults to 1 sec.

    *  - ``progressive``
       - ``bool``
       - When optimizing patterns for a scattering medium, it is useful to run
         the first few iterations with scattering disabled, and then enable it.
         This flag enables this option. Defaults to False.

    *  - ``surface_aware``
       - ``bool``
       - Determines whether our surface-aware discretization should be used, or
         a simple discretization to a binary occupancy grid instead. Defaults to
         False.

    *  - ``filter_radon``
       - ``bool``
       - If enabled, the Radon transform of the target object will first be
         computed, and then all projector pixels where it has value 0 will be
         disabled. This can significantly speed up the optimization for objects
         not covering the entire projection surface. Defaults to False.

    *  - ``output``
       - ``str``
       - The output directory where the results will be saved. If not specified,
         the results will be saved in the directory containing this file.

Limitations
-----------

This command-line interface is meant to be a simple way to run an optimization,
and therefore allows limited flexibility regarding the possible optimizations.
For more advanced usage, we recommend setting your own optimization pipeline
using the Python API directly. Please refer to the :ref:`tutorials` section for
more information.

