.. drtvam documentation master file, created by
   sphinx-quickstart on Sun Nov 17 18:02:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dr.TVAM
=======

`Dr.TVAM <https://github.com/rgl-epfl/drtvam>`_ is an inverse rendering framework for tomographic additive
manufacturing. It provides a set of tools to model a TVAM printing process and
optimize projection patterns for it. It is implemented in Python on top of the
`Mitsuba 3 <https://github.com/mitsuba-renderer/mitsuba3>`_ framework.

It is possible to use Dr.TVAM to setup a wide variety of experiments. In
particular, we support:

* Printing in scattering media. 
* Printing with arbitrary container geometry (round, square, ...), or
  occluding objects (e.g. overprinting), with correct handling of the light
  transport. 
* Printing with a variety of projection models, and an arbitrary
  container motion. 
* Optimizing patterns for TVAM with our surface-aware
  discretization.
* High-performance, see also :ref:`performance_tips` 

Installation
------------

We recommend installing Dr.TVAM directly with ``pip``:


.. code-block:: bash

    pip install drtvam

Or the current development version with ``pip``:

.. code-block:: bash

    pip install https://github.com/rgl-epfl/drtvam

Alternatively, you can clone the repository and install it manually:

.. code-block:: bash

    git clone --recursive git@github.com:rgl-epfl/drtvam.git
    pip install ./drtvam

Requirements
^^^^^^^^^^^^

- ``Python >= 3.8``
- A NVIDIA GPU with driver version ``>= 495.89``
- Alternatively, to use Dr.TVAM on the CPU: ``LLVM >= 11.1``. The performance
  will be significantly lower than on the GPU, however.

Getting started
---------------

Dr.TVAM is a python-based framework that allows great flexibility in setting up
an inverse optimization for TVAM. We also provide a command-line tool to run an
optimization from a configuration file. We detail the how to use it in the
:ref:`basic_usage` section.

For more advanced usage of Dr.TVAM, it is possible to implement one's own
optimization pipeline. We provide a set of tutorials to explain the key concepts
in the :ref:`tutorials` section.

License
-------
Dr.TVAM is provided as free, open-source software under a non-commercial
licence. It may be used freely for academic purposes, but should not be used
under any circumstances for commercial purposes.


Citation
--------

When using Dr.TVAM in academic projects, please cite the associated publications:

.. code-block:: bibtex

    @article{nicolet2024inverse,
        author = {Nicolet, Baptiste and Wechsler, Felix and Madrid-Wolff, Jorge and Moser, Christophe and Jakob, Wenzel},
        title = {Inverse Rendering for Tomographic Volumetric Additive Manufacturing},
        journal = {Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
        volume = {43},
        number={6},
        year = {2024},
        month = {12},
        doi = {10.1145/3687924}
    }

    @misc{wechsler_2025_overprinting_TVAM,
      title={Overprinting with Tomographic Volumetric Additive Manufacturing}, 
      author={Felix Wechsler and Viola Sgarminato and Riccardo Rizzo and Baptiste Nicolet and Wenzel Jakob and Christophe Moser},
      year={2025},
      eprint={2507.13842},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2507.13842}, 
    }

.. .............................................................................

.. toctree::
    :maxdepth: 1
    :hidden:

    basic_usage
    tutorials
    plugin_doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
