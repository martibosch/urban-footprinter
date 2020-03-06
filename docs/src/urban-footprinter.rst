=================
Urban footprinter
=================

The urban footprinter package provides two ways to execute the procedure described above. On the one hand, an *object-oriented* interface which is well suited for the interactive exploration with different parameter values. On the other hand, the whole procedure is also encapsulated into a single function.

------------------------
Object-oriented approach
------------------------

The main advantage of using the `UrbanFootprinter` class, is that the convolution results (for a given `kernel_radius`) are cached, so that further calls avoid performing it if it is not necessary.

.. autoclass:: urban_footprinter.UrbanFootprinter
    :members:  __init__, compute_footprint_mask, compute_footprint_mask_shp

------------------------
Single function approach
------------------------

If interactive exploration of the parameters is not required, the whole procedure described above can be executed with the single function named `urban_footprint_mask`, which accepts the following arguments:

.. autofunction:: urban_footprinter.urban_footprint_mask

The urban extent can also be obtained as a vector geometry (instead of a raster array) by means of the `urban_footprint_mask_shp` function:

.. autofunction:: urban_footprinter.urban_footprint_mask
