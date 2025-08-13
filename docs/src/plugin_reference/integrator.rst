.. _integrator:

Integrators
===========

The integrator is the element that implements the *rendering* operation, i.e.
the forward simulation of the TVAM process. It start light paths on the
projector, and computes their propagation in the scene until they reach the
sensor, and it then accumulates absoption on the sensor's film.

Volume integrator (``volume``)
------------------------------

The main integrator that should be used to model the TVAM process and for
optimization is the ``volume`` integrator. It is a custom integrator that is
specifically designed to compute the absorption map in the volumetric film. 

It can be loaded as a Mitsuba plugin:

.. code-block:: python

    integrator = mi.load_dict({
        'type': 'volumeintegrator',
        'print_time': 20.0,
    })

It takes the following parameters:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``sample_time``
      - ``bool``
      - Flag that specifies if the time dimension should be sampled
        stochastically or not. If ``True``, the integrator will generate samples
        from each projector pixel at random angles, which allows to account for
        motion blur. Otherwise, each path will start at the discrete angles
        determined by the number of patterns in one rotation. Defaults to
        ``False``.

    * - ``regular_sampling``
      - ``bool``
      - Whether light paths should start from the center of each projector pixel
        or not. By default, paths start from a random position within one pixel.
        Defaults to ``False``.

    * - ``print_time``
      - ``float``
      - Total printing time, in seconds. This defines the total exposure time.
        Defaults to ``1.0``.

    * - ``target_id``
      - ``str``
      - The ID of the target shape for an optimization. This is useful in
        certain cases where it is necessary during the rendering phase to know
        if the shape that a light path is interacting with is the target or not.
        Defaults to 'target'.

    * - ``transmission_only``
      - ``bool``
      - Flag that specifies if the integrator should only evaluate the
        transmission component of the BSDF at each interaction. This allows to
        force all paths to enter the medium while still correctly accounting for
        attenuation at the interfaces. Defaults to ``True``.


Radon integrator (``radon``)
----------------------------

For convenience, we also provide an integrator that computes the Radon transform
of the target object. It accepts the same parameters as the volume integrator.

.. warning::
   This integrator is a *forward-only* integrator, and cannot be used for
   optimization. It is only useful to compute the Radon transform of a shape.

