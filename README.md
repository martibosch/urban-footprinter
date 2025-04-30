[![PyPI version fury.io](https://badge.fury.io/py/urban-footprinter.svg)](https://pypi.python.org/pypi/urban-footprinter/)
[![Documentation Status](https://readthedocs.org/projects/urban-footprinter/badge/?version=latest)](https://urban-footprinter.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/urban-footprinter/actions/workflows/tests.yml/badge.svg)](https://github.com/martibosch/urban-footprinter/blob/main/.github/workflows/tests.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/urban-footprinter/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/urban-footprinter/main)
[![codecov](https://codecov.io/gh/martibosch/urban-footprinter/branch/main/graph/badge.svg?token=H8PW6I8DY5)](https://codecov.io/gh/martibosch/urban-footprinter)
[![GitHub license](https://img.shields.io/github/license/martibosch/urban-footprinter.svg)](https://github.com/martibosch/urban-footprinter/blob/main/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martibosch/urban-footprinter/main?filepath=notebooks/overview.ipynb)
[![DOI](https://zenodo.org/badge/215518909.svg)](https://zenodo.org/badge/latestdoi/215518909)

# Urban footprinter

A reusable convolution-based approach to detect urban extents from raster datasets.

|                    LULC                    |                       Convolution result                        |                Computed urban extent                 |
| :----------------------------------------: | :-------------------------------------------------------------: | :--------------------------------------------------: |
| ![LULC](notebooks/figures/zurich-lulc.png) | ![Convolution result](notebooks/figures/zurich-conv-result.png) | ![Urban extent](notebooks/figures/zurich-extent.png) |

The approach is built upon the methods used in the [Atlas of Urban Expansion](http://atlasofurbanexpansion.org/). The main idea is that a pixel is considered part of the urban extent depending on the proportion of built-up pixels that surround it. See the [notebook overview](https://github.com/martibosch/urban-footprinter/tree/main/notebooks/overview.ipynb) or [this blog post](https://martibosch.github.io/urban-footprinter/) for a more detailed description of the procedure.

**Citation**: Bosch M. 2020. "Urban footprinter: a convolution-based approach to detect urban extents from raster data". Available from [https://github.com/martibosch/urban-footprinter](https://github.com/martibosch/urban-footprinter). Accessed: DD Month YYYY.

An example BibTeX entry is:

```bibtex
@misc{bosch2020urban,
  title={Urban footprinter: a convolution-based approach to detect urban extents from raster data},
  author={Bosch, Mart\'{i}},
  year={2020},
  doi={10.5281/zenodo.3699310},
  howpublished={Available from \url{https://github.com/martibosch/urban-footprinter}. Accessed: DD Month YYYY},
}
```

## Installation and usage

To install use pip:

```
$ pip install urban-footprinter
```

Then use it as:

```python
import urban_footprinter as ufp

# Or use `ufp.urban_footprint_mask_shp` to obtain the urban extent as a
# shapely geometry
urban_mask = ufp.urban_footprint_mask(
    "path/to/raster.tif", kernel_radius, urban_threshold, urban_classes=urban_classes
)
```

where

```
help(ufp.urban_footprint_mask)

Help on function urban_footprint_mask in module urban_footprinter:

urban_footprint_mask(raster, kernel_radius, urban_threshold, urban_classes=None, num_patches=1,
                     buffer_dist=None, res=None)
    Computes a boolean mask of the urban footprint of a given raster.

    Parameters
    ----------
    raster : ndarray or str, file object or pathlib.Path object
        Land use/land cover (LULC) raster. If passing a ndarray (instead of the
        path to a geotiff), the resolution (in meters) must be passed to the
        `res` keyword argument.
    kernel_radius : numeric
        The radius (in meters) of the circular kernel used in the convolution.
    urban_threshold : float from 0 to 1
        Proportion of neighboring (within the kernel) urban pixels after which
        a given pixel is considered urban.
    urban_classes : int or list-like, optional
        Code or codes of the LULC classes that must be considered urban. Not
        needed if `raster` is already a boolean array of urban/non-urban LULC
        classes.
    num_patches : int, default 1
        The number of urban patches that should be featured in the returned
        urban/non-urban mask. If `None` or a value lower than one is provided,
        the returned urban/non-urban mask will features all the urban patches.
    buffer_dist : numeric, optional
        Distance to be buffered around the urban/non-urban mask. If no value is
        provided, no buffer is applied.
    res : numeric, optional
        Resolution of the `raster` (assumes square pixels). Ignored if `raster`
        is a path to a geotiff.

    Returns
    -------
    urban_mask : ndarray
```
