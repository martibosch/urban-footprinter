[![PyPI version fury.io](https://badge.fury.io/py/urban-footprinter.svg)](https://pypi.python.org/pypi/urban-footprinter/)
[![Build Status](https://travis-ci.org/martibosch/urban-footprinter.svg?branch=master)](https://travis-ci.org/martibosch/urban-footprinter)
[![Coverage Status](https://coveralls.io/repos/github/martibosch/urban-footprinter/badge.svg?branch=master)](https://coveralls.io/github/martibosch/urban-footprinter?branch=master)
[![GitHub license](https://img.shields.io/github/license/martibosch/urban-footprinter.svg)](https://github.com/martibosch/urban-footprinter/blob/master/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martibosch/urban-footprinter/master?filepath=notebooks/overview.ipynb)

Urban footprinter
===============================

Overview
--------

A convolution-based approach to detect urban extents from raster datasets.

LULC | Convolution result | Computed urban extent
:-------------------------:|:-------------------------:|:-------------------------:
![LULC](notebooks/figures/zurich-lulc.png) | ![Convolution result](notebooks/figures/zurich-conv-result.png) | ![Urban extent](notebooks/figures/zurich-extent.png)

See the [notebook overview](https://github.com/martibosch/urban-footprinter/tree/master/notebooks/overview.ipynb) for more details on the procedure.

Installation
------------

To install use pip:

    $ pip install urban-footprinter


Or clone the repo:

    $ git clone https://github.com/martibosch/urban-footprinter.git
    $ python setup.py install
