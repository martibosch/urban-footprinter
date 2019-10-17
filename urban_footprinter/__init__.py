import numpy as np
import rasterio as rio
from rasterio import features
from scipy import ndimage as ndi
from shapely import geometry

__version__ = '0.0.1'

__all__ = ['urban_footprint_mask', 'urban_footprint_mask_shp']

_urban_footprint_mask_doc = """
Computes a boolean mask of the urban footprint of a given raster.

Parameters
----------
raster : ndarray or str, file object or pathlib.Path object
    Land use/land cover (LULC) raster. If passing a ndarray (instead of the
    path to a geotiff), the resolution (in meters) must be passed to the `res`
    keyword argument.
kernel_radius : numeric
    The radius (in meters) of the circular kernel used in the convolution.
urban_threshold : float from 0 to 1
    Proportion of neighboring (within the kernel) urban pixels after which a
    given pixel is considered urban.
urban_classes : int or list-like, optional
    Code or codes of the LULC classes that must be considered urban. Not
    needed if `raster` is already a boolean array of urban/non-urban LULC
    classes.
largest_patch_only : boolean, default True
    Whether the returned urban/non-urban mask should feature only the largest
    urban patch.
buffer_dist : numeric, optional
    Distance to be buffered around the urban/non-urban mask. If no value is
    provided, no buffer is applied.
%s
Returns
-------
urban_mask : %s
 """


def urban_footprint_mask(raster, kernel_radius, urban_threshold,
                         urban_classes=None, largest_patch_only=True,
                         buffer_dist=None, res=None):
    """
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
    largest_patch_only : boolean, default True
        Whether the returned urban/non-urban mask should feature only the
        largest urban patch.
    buffer_dist : numeric, optional
        Distance to be buffered around the urban/non-urban mask. If no value is
        provided, no buffer is applied.
    res : numeric, optional
        Resolution of the `raster` (assumes square pixels). Ignored if `raster`
        is a path to a geotiff.

    Returns
    -------
    urban_mask : ndarray
    """

    if isinstance(raster, np.ndarray):
        if res is None:
            raise ValueError("Must provide `res` if raster is a ndarray")
    else:
        with rio.open(raster) as src:
            raster = src.read(1)
            if res is None:
                res = src.res[0]  # only square pixels are supported

    if urban_classes is None:
        # no need to use `np.copy` because of the `astype` below
        urban_lulc_arr = raster
    if isinstance(urban_classes, (list, tuple)):
        urban_lulc_arr = np.isin(raster, urban_classes)
    else:
        urban_lulc_arr = raster == urban_classes
    urban_lulc_arr = urban_lulc_arr.astype(np.uint32)

    # get circular kernel for the convolution
    kernel_pixel_radius = int(kernel_radius // res)
    kernel_pixel_len = 2 * kernel_pixel_radius + 1

    y, x = np.ogrid[-kernel_pixel_radius:kernel_pixel_len -
                    kernel_pixel_radius, -kernel_pixel_radius:
                    kernel_pixel_len - kernel_pixel_radius]
    mask = x * x + y * y <= kernel_pixel_radius * kernel_pixel_radius

    kernel = np.zeros((kernel_pixel_len, kernel_pixel_len), dtype=np.uint32)
    kernel[mask] = 1

    # conv_result = ndi.convolve(raster, kernel)
    # kernel_pixel_area = kernel_pixel_len * kernel_pixel_len - 1
    urban_mask = ndi.convolve(urban_lulc_arr, kernel) >= \
        urban_threshold * (kernel_pixel_len * kernel_pixel_len - 1)

    if largest_patch_only:
        # use the connected-component labelling to extract the largest
        # contiguous urban patch
        kernel_moore = ndi.generate_binary_structure(2, 2)
        label_arr = ndi.label(urban_mask, kernel_moore)[0]
        cluster_label = np.argmax(
            np.unique(label_arr, return_counts=True)[1][1:]) + 1
        urban_mask = (label_arr == cluster_label).astype(np.uint8)

    if buffer_dist is not None:
        iterations = int(buffer_dist // res)
        kernel_moore = ndi.generate_binary_structure(2, 2)
        urban_mask = ndi.binary_dilation(urban_mask, kernel_moore,
                                         iterations=iterations)

    return urban_mask


def urban_footprint_mask_shp(raster, kernel_radius, urban_threshold,
                             urban_classes=None, largest_patch_only=True,
                             buffer_dist=None):
    """
    Computes a geometry of the urban footprint of a given raster.

    Parameters
    ----------
    raster : str, file object or pathlib.Path object
        Path to the land use/land cover (LULC) raster.
    kernel_radius : numeric
        The radius (in meters) of the circular kernel used in the convolution.
    urban_threshold : float from 0 to 1
        Proportion of neighboring (within the kernel) urban pixels after which
        a given pixel is considered urban.
    urban_classes : int or list-like, optional
        Code or codes of the LULC classes that must be considered urban. Not
        needed if `raster` is already a boolean array of urban/non-urban LULC
        classes.
    largest_patch_only : boolean, default True
        Whether the returned urban/non-urban mask should feature only the
        largest urban patch.
    buffer_dist : numeric, optional
        Distance to be buffered around the urban/non-urban mask. If no value is
        provided, no buffer is applied.

    Returns
    -------
    urban_mask_geom : geometry
    """

    with rio.open(raster) as src:
        raster = src.read(1)

        urban_mask = urban_footprint_mask(
            src.read(1), kernel_radius, urban_threshold,
            urban_classes=urban_classes, largest_patch_only=largest_patch_only,
            buffer_dist=buffer_dist, res=src.res[0])

        # if geometry is None:
        #     logging.warning(
        #         "Building the mask vector geometry requires the shapely "
        #         "package. Returning raster urban mask instead")
        #     return urban_mask

        return geometry.shape([(geom, val) for geom, val in features.shapes(
            urban_mask, transform=src.transform) if val == 1][-1][0])
