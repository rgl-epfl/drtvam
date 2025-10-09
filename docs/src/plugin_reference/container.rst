.. 03: 

Containers
==========

Containers in Dr.TVAM are a useful abstraction to define the resin and its
container. Unlike other plugins, it is only a convenience class so that users do
not need to be concerned with the nitty-gritty details of setting up surfaces
and materials in a Mitsuba 3 scene.

Containers are implemented as child classes of the base ``Container`` class, and
they implement one method ``to_dict``, which returns a Mitsuba-compatible
dictionary that can be used to define the container in a Mitsuba 3 scene.

The container class serves two purposes:

1. It defines the printing medium and its optical properties.
2. It defines the container geometry and its optical properties.

A container can be instantiated as follows:

.. code-block:: python

    vial = IndexMatchedVial({
        'r': 2.5,
        'height': 10,
        'medium': {
            'extinction': 0.1,
            'albedo': 0.9,
            'phase': {
                'type': 'hg',
                'g': 0.5
            },
            'ior': 1.5
        }
    })

We explain the parameters of the constructor in the following sections. After
instantiating the container, the Mitsuba-compatible dictionary can be obtained
by calling the ``to_dict`` method, e.g. to add it to an existing Mitsuba scene
dictionary under construction:

.. code-block:: python

    scene_dict |= vial.to_dict()


Printing medium
---------------

The printing medium is characterized by a few aspects:

* its absorbtion and scattering coefficients, which respectively quantify the
  probability of a photon being absorbed or scattered along a ray per unit
  length. These coefficients are summed to form the extinction coefficient,
  which describes the total attenuation of light along a ray in the medium. The
  ratio of scattering to extinction is called the *single-scattering albedo*.
* its phase function, which defines the probability of a photon being scattered
  in a given direction given its incoming direction.
* its refractive index, which defines the speed of light in the medium.

Defining a printing medium in Dr.TVAM is done by specifying each of these
properties in the ``medium`` dictionary of the container class. The medium
dictionary should contain the following entries:


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``extinction``
      - ``float``
      - The extinction coefficient of the medium, in (scene units)^-1.

    * - ``albedo``
      - ``float``
      - The single-scatterig albedo of the medium, in [0, 1]. A value of 0
        indicates that the medium is purely absorptive.

    * - ``phase``
      - ``dict``
      - Mitsuba-compatible dictionary defining the phase function of the medium.
        See the corresponding `section
        <https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html>`_
        in the Mitsuba 3 documentation for more information.

    * - ``ior``
      - ``float``
      - The refractive index of the medium.



Container geometry
------------------

The container geometry is defined by a set of *surfaces* that enclose the resin.
Each surface is defined by a Mitsuba `shape plugin
<https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html>`_.
The ``Container`` class creates the dictionary representation of the container
by combining the surfaces and the printing medium. A few containers are
supported:

Index-matched vial (``index_matched``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This container implements an idealized version of a printing setup with a
cylindrical vial immersed in an index-matching bath. In that case, all
reflections and refractions at the interfaces are ignored, as if the vial was a
perfectly transparent cylinder. This is what this plugin implements. It takes
the following parameters:


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``r``
      - ``float``
      - The radius of the vial, in scene units.

    * - ``height``
      - ``float``
      - The height of the vial, in scene units.

    * - ``medium``
      - ``dict``
      - The medium dictionary, as described above.

    * - ``ior_surrounding``
      - ``float``
      - (Optional) The refractive index of the surrounding medium. If not
        specified, it is assumed to be 1.0 (air). 

Cylindrical vial (``cylindrical``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This container implements a cylindrical vial, with no index-matching bath. The
system correctly accounts for the attenuation and change of direction at the
dielectric interfaces. It takes the following parameters:


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``r_int``
      - ``float``
      - The interior radius of the vial, in scene units.

    * - ``r_ext``
      - ``float``
      - The exterior radius of the vial, in scene units.

    * - ``height``
      - ``float``
      - The height of the vial, in scene units.

    * - ``ior``
      - ``float``
      - The refractive index of the vial.

    * - ``medium``
      - ``dict``
      - The medium dictionary, as described above.
    
    * - ``ior_surrounding``
      - ``float``
      - (Optional) The refractive index of the surrounding medium. If not
        specified, it is assumed to be 1.0 (air). 

Square vial (``square``)
^^^^^^^^^^^^^^^^^^^^^^^^

This container implements a vial with a square cross-section, like a
spectroscopy cuvette. It takes the following parameters:


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``w_int``
      - ``float``
      - The interior length of one side of the vial, in scene units.

    * - ``w_ext``
      - ``float``
      - The exterior length of one side of the vial, in scene units.

    * - ``height``
      - ``float``
      - The height of the vial, in scene units.

    * - ``ior``
      - ``float``
      - The refractive index of the vial.

    * - ``medium``
      - ``dict``
      - The medium dictionary, as described above.

    * - ``ior_surrounding``
      - ``float``
      - (Optional) The refractive index of the surrounding medium. If not
        specified, it is assumed to be 1.0 (air). 




Double Cylindrical vial (``double_cylindrical``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This container implements a cylindrical vial (outer vial) which contains the medium, with no index-matching bath. 
Inside that medium is a smaller cylinder (inner vial) placed which itself is filled with another medium.
The system correctly accounts for the attenuation and change of direction at the
dielectric interfaces. It takes the following parameters:


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``r_int_outer``
      - ``float``
      - The interior radius of the outer vial, in scene units.

    * - ``r_ext_outer``
      - ``float``
      - The exterior radius of the outer vial, in scene units.

    * - ``r_int_inner``
      - ``float``
      - The interior radius of the inner vial, in scene units.

    * - ``r_ext_inner``
      - ``float``
      - The exterior radius of the inner vial, in scene units.

    * - ``ior_outer``
      - ``float``
      - The refractive index of the outer vial.

    * - ``ior_inner``
      - ``float``
      - The refractive index of the inner vial.

    * - ``ior_inside_inner``
      - ``float``
      - The refractive index of the medium inside the inner vial.

    * - ``medium``
      - ``dict``
      - The medium dictionary, as described above.

    * - ``ior_surrounding``
      - ``float``
      - (Optional) The refractive index of the surrounding medium. If not
        specified, it is assumed to be 1.0 (air). 


Custom vial (``custom``)
^^^^^^^^^^^^^^^^^^^^^^^^

This container implements a custom shaped vial which
is specified with two mesh files which define the interior and exterior.
Be sure that both mesh normals point outward.


.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``filename_vial_outer``
      - ``str``
      - The path to the mesh file defining the exterior of the vial

    * - ``filename_vial_inner``
      - ``str``
      - The path to the mesh file defining the interior of the vial

    * - ``ior``
      - ``float``
      - The refractive index of the vial.

    * - ``medium``
      - ``dict``
      - The medium dictionary, as described above.

    * - ``ior_surrounding``
      - ``float``
      - (Optional) The refractive index of the surrounding medium. If not
        specified, it is assumed to be 1.0 (air). 



Occlusions
----------
We support one or several occlusions inside the medium which can be fully absorptive, transparent or scattering.
In fact we support all `Bidirectional Scattering Distribution Functions (BSDFs) <https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html>`_ that Mitsuba 3 supports.
The occlusions are defined in the ``occlusions`` list of the container dictionary. By default the occlusions are fully absorptive, but you can specify a different BSDF for each occlusion.


Fully absorptive occlusion
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: json

    "vial": {
        "type": "cylindrical",
        "r_int": 6.363125,
        "r_ext": 7.354374999999999,
        "ior": 1.54,
        "medium": {
            "ior": 1.4849,
            "phase": {"type": "rayleigh"},
            "extinction": 0.1450628608586249,
            "albedo": 0.0
        },
        "occlusions": [
            {
                "filename": "fully_absorptive.ply"
            }
        ]
    },

BSDF specified occlusion
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: json

    "vial": {
        "type": "square",
        "w_int": 10.191,
        "w_ext": 12.408,
        "ior": 1.54,
        "medium": {
            "ior": 1.3512,
            "phase": {
                "type": "hg",
                "g": 0.95
            },
            "extinction": 0.036657,
            "albedo": 0.0
        },
        "occlusions": [
            {
                "filename": "plastic_piece.ply",
                "bsdf": {
                    "type": "dielectric",
                    "int_ior": 1.58,
                    "ext_ior": 1.4849
                }
            },
            {
                "filename":"fully_absorptive.ply"
            }
        ]
    }
