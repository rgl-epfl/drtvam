Release notes
=============

Being an experimental research framework, Dr.TVAM does not strictly follow the
`Semantic Versioning <https://semver.org/>`_ convention. That said, we will
strive to document breaking API changes in the release notes below.


Dr. TVAM 0.5.0
--------------
*August 13, 2025*

- Introduce support for Mitusba 3.7.0 `[#50] <https://github.com/rgl-epfl/drtvam/pull/50>`_

Dr. TVAM 0.4.0
--------------
*August 6, 2025*

- Change the intensity normalization of lens projector `[#47] <https://github.com/rgl-epfl/drtvam/pull/47>`_. It uses now the same normalization as the telecentric and collimated projectors.
- Introduce forward mode to just project patterns instead of optimizing `[#46] <https://github.com/rgl-epfl/drtvam/pull/46>`_. For example: ``drtvam new_conditions/config.json --forward_mode --patterns old_conditions/patterns.npz`

Dr.TVAM 0.3.1
-------------
*August 4, 2025*

- Add `telecentric` imaging system as projector `[#39] <https://github.com/rgl-epfl/drtvam/pull/39>`_
- Add option to trace single pixels `[#39] <https://github.com/rgl-epfl/drtvam/pull/39>`_


Dr.TVAM 0.3.0
-------------
*July 11, 2025*

- Make occlusions more generic `[#27] <https://github.com/rgl-epfl/drtvam/pull/27>`_
- Introduce a generic cuvette via meshes `[#30] <https://github.com/rgl-epfl/drtvam/pull/30>`_
- Add option to move target file `[#26] <https://github.com/rgl-epfl/drtvam/pull/26>`_
- Minor fixes for surface aware discretization
- Some fixes for filter radon
- Improved histogram export
- Add double cyclindrical vials `[#37] <https://github.com/rgl-epfl/drtvam/pull/37>`_ 

Dr.TVAM 0.2.0
-------------
*March 7, 2025*

- Fixes to surface-aware discretization `[a3c6430] <https://github.com/rgl-epfl/drtvam/commit/a3c64302f78b3694fd65dd7cc683f852c2a8cb33>`_
- Adds a sparsity loss and weights to all loss terms `[b1f9c33] <https://github.com/rgl-epfl/drtvam/commit/b1f9c33a5d319157972711f224451cbab4a9beb1>`_
- Improved documentation `[29e055d] <https://github.com/rgl-epfl/drtvam/commit/29e055db98ee1ca18a4d051a61f403c64696fe19>`_

Dr.TVAM 0.1.0
-------------
*November 26, 2024*

- Initial release
