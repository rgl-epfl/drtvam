<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="https://rgl.epfl.ch/publications/Nicolet2024Inverse">Dr.TVAM</a></h1>

  <a href="https://rgl.epfl.ch/publications/Nicolet2024Inverse">
    <img src="https://rgl.s3.eu-central-1.amazonaws.com/media/images/papers/Nicolet2024Inverse_teaser_1.png" alt="Logo" width="100%">
  </a>

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia), December 2024.
    <br />
    <a href="https://bnicolet.com/"><strong>Baptiste Nicolet</strong></a>
    ·
    <a href="https://www.felixwechsler.science/"><strong>Felix Wechsler</strong></a>
    ·
    <a href="https://www.linkedin.com/in/jorge-madrid-wolff/"><strong>Jorge Madrid-Wolff</strong></a>
    ·
    <a href="https://www.epfl.ch/labs/lapd/page-67957-en-html/"><strong>Christophe Moser</strong></a>
    ·
    <a href="https://rgl.epfl.ch/people/wjakob"><strong>Wenzel Jakob</strong></a>
  </p>

  <p align="center">
    <a href='https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2024Inverse.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://rgl.epfl.ch/publications/Nicolet2024Inverse' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
    <a href='https://drtvam.readthedocs.io/en/latest/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Docs-Page-blue?style=flat-square' alt='Project Documentation'>
    </a>
  </p>
</p>

<p align="center">
  <a href='https://rgl-ci.epfl.ch/viewType.html?buildTypeId=DrTVAM_LinuxAmd64gcc9&guest=1'>
    <img src='https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:DrTVAM_LinuxAmd64gcc9)/statusIcon.svg'>
  </a>
</p>

<br />
<br />

## Abstract
Tomographic Volumetric Additive Manufacturing (TVAM) is an emerging 3D printing technology that can create complex objects in under a minute. The key idea is to project intense light patterns onto a rotating vial of photo-sensitive resin, causing polymerization where the cumulative dose of these patterns reaches the polymerization threshold. We formulate the pattern calculation as an inverse light transport problem and solve it via physically based differentiable rendering. In doing so, we address long-standing limitations of prior work by accurately modeling and correcting for scattering in composite resins, printing in non-symmetric vials, and supporting unusual printing geometries. We also introduce an improved discretization scheme that exploits the ray tracing operation to mitigate resolution-related artifacts in prints. We demonstrate the benefits of our method in real-world experiments, where our computed patterns produce prints with an improved fidelity.



## About this project

Dr.TVAM is a high-performance inverse rendering framework for tomographic volumetric additive
manufacturing. It is based on the [Mitsuba renderer](https://www.mitsuba-renderer.org/), and uses physically-based
differentiable rendering to optimize patterns for TVAM. In particular, it supports:

- Printing in scattering media
- Arbitrary vial shapes (round, square, ...)
- Arbitrary projector motions (orthogonal, tilted)
- An improved discretization scheme for the target shape to reduce computational load and to minimize discretization artifacts

For more details we refer to this [publication](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2024Inverse.pdf).


## Overprinting
We have recently extended this project to describe also a variety of overprinting scenarios. See [Overprinting with Tomographic Volumetric Additive Manufacturing](https://github.com/EPFL-LAPD/Overprinting-with-Tomographic-Volumetric-Additive-Manufacturing).

## ⚠️ Sign up for online workshop in late summer 2025
* The date is not announced yet, but sign up for the announcement: [here](https://forms.office.com/e/3E0SeYUR1X?origin=lprLink)

## Installation

Installing Dr.TVAM can be done via `pip`:

```bash=
pip install drtvam
```
or the current development version
```bash=
pip install git+https://github.com/rgl-epfl/drtvam
```

## Basic Usage

We provide a convenience command-line tool `drtvam` to run simple optimizations. You can run it as:

```bash=
drtvam path/to/config.json
```
Please refer to the documentation for details on the configuration file format. 
Dr.TVAM will run multi-threaded on your machine but will also use your CUDA GPU and your [RT cores](https://developer.nvidia.com/rtx/ray-tracing) if supported by your hardware. 

## Advanced Usage

Dr.TVAM provides a set of useful abstractions to implement a wide variety of
custom TVAM setups. We show examples in the documentation to get you started.

## Documentation

The full documentation for this project, along with jupyter notebooks
explaining the basics of implementing your own optimizations in our framework,
can be found on [readthedocs](https://drtvam.readthedocs.io/en/latest/).

## Benchmarks
We have done some simple benchmarks: 
* **RTX3060 12GB (2020)**: Entry-level consumer hardware GPU
* **L40S 48GB (2022)**: Professional graphics card used primarily in server clusters

### Test Subjects

* **Dr. TVAM**: Tested on **RTX3060 12GB** and **L40S 48GB**.
* **Benchy** boat as target:
  * Size: **10mm**
  * Pixel size: **25µm**
  * Resolution on DMD: **400x400 pixels**
  * Angles: **400**
  * 40 iterations with gradient-based L-BFGS optimizer

### Computation Performance

| Configuration | Rays per Pixel |  Time (RTX3060) |  Time (L40S) |
| --- | --- | --- | --- |
| Julia Radon based | 1 | **0h 3m 17s** |  |
| Dr. TVAM index-matched | 1 | **0h 2m 2s** | **0h 0m 20s** |
| Dr. TVAM index-matched | 1 | **0h 2m 2s** | **0h 0m 20s** |
| Dr. TVAM cylindrical | 1 | **0h 2m 10s** | **0h 0m 23s** |
| Dr. TVAM square | 1 | **0h 2m 15s** | **0h 0m 20s** |
| Dr. TVAM cylindrical scattering | 16 | **1h 40m 0s** | **0h 14m 10s** |
| Dr. TVAM square scattering | 16 | **1h 40m 0s** | **0h 14m 20s** |
| Dr. TVAM square scattering (surface-aware loss, disable black pixels) | 16 | **0h 25m 0s** | **0h 3m 45s** |



## Teaser video
Can be found on YouTube:

<a  href="https://www.youtube.com/watch?v=w6weg4noMIY"><img src="docs/src/resources/video_teaser.png"  width="400"></a>

## Issues or support needed
In case you run into issues or you do need support, do not hesitate to [open an issue](https://github.com/rgl-epfl/drtvam/issues/new) such that we can help you using Dr. TVAM!
As an academic user, this is completely free to use. Please reach out to us in case you need support!

## License

This project is provided under a non-commercial license. Please refer to the LICENSE file for details.

## Citation

When using this project in academic works, please cite the following paper:

```
@article{nicolet2024inverse,
    author = {Nicolet, Baptiste and Wechsler, Felix and Madrid-Wolff, Jorge and Moser, Christophe and Jakob, Wenzel},
    title = {Inverse Rendering for Tomographic Volumetric Additive Manufacturing},
    journal = {Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
    volume = {43},
    number={6},
    year = {2024},
    month = dec,
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
```


