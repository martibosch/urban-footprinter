"""Urban footprinter."""

import cv2
import numpy as np
import rasterio as rio
from pandas.api.types import is_list_like
from rasterio import features
from scipy import ndimage as ndi
from shapely import geometry

from urban_footprinter import settings

__version__ = "0.3.1"

__all__ = [
    "UrbanFootprinter",
    "urban_footprint_mask",
    "urban_footprint_mask_shp",
]

KERNEL_MOORE = ndi.generate_binary_structure(2, 2)


class UrbanFootprinter:
    """Urban footprinter."""

    def __init__(
        self,
        raster,
        urban_classes=None,
        res=None,
        convolve_border_type=None,
        lulc_dtype=None,
        mask_dtype=None,
    ):
        """Initialize the urban footprinter.

        Parameters
        ----------
        raster : ndarray or str, file object or pathlib.Path object
            Land use/land cover (LULC) raster. If passing a ndarray (instead of the path
            to a geotiff), the resolution (in meters) must be passed to the `res`
            keyword argument.
        urban_classes : int or list-like, optional
            Code or codes of the LULC classes that must be considered urban. Not needed
            if `raster` is already a boolean array of urban/non-urban LULC classes.
        res : numeric, optional
            Resolution of the `raster` (assumes square pixels). Ignored if `raster` is a
            path to a geotiff.
        convolve_border_type : int, optional
            The type of border to use when convolving the raster with the kernel. Must
            be an integer corresponding to an opencv border type. See the opencv docs
            for a list of possible values. If not provided, the default value from
            `settings.DEFAULT_CONV_BORDER_TYPE` is used.
        lulc_dtype : str or numpy dtype, optional
            Data type to be used for the LULC array. It may need to be higher than the
            raster's original data type to avoid integer overflow when convolving the
            LULC array. If not provided, the default value from
            `settings.DEFAULT_LULC_DTYPE` is used.
        mask_dtype : str or numpy dtype, optional
            Data type to be used for the returned mask. If not provided, the default
            value from `settings.DEFAULT_MASK_DTYPE` is used.

        """
        if isinstance(raster, np.ndarray):
            if res is None:
                raise ValueError("Must provide `res` if raster is a ndarray")
        else:
            with rio.open(raster) as src:
                raster = src.read(1)
                res = src.res[0]  # only square pixels are supported
                self.transform = src.transform
        self.res = res
        if convolve_border_type is None:
            convolve_border_type = settings.DEFAULT_CONV_BORDER_TYPE
        self.convolve_border_type = convolve_border_type
        if lulc_dtype is None:
            lulc_dtype = settings.DEFAULT_LULC_DTYPE
        self.lulc_dtype = lulc_dtype
        if mask_dtype is None:
            mask_dtype = settings.DEFAULT_MASK_DTYPE
        self.mask_dtype = mask_dtype

        if urban_classes is None:
            # no need to use `np.copy` because of the `astype` below
            urban_lulc_arr = raster
        if is_list_like(urban_classes):
            urban_lulc_arr = np.isin(raster, urban_classes)
        else:
            urban_lulc_arr = raster == urban_classes

        # need to ensure a dtype that allows for high positive integers
        # (convolution results can be quite high)
        self.urban_lulc_arr = urban_lulc_arr.astype(self.lulc_dtype)

        # init an empty dict to cache the convolution results of each kernel
        # radius
        self._convolution_result_dict = {}

    def get_convolution_result(self, kernel_radius):
        """Get the result of convolving the raster with the kernel.

        Parameters
        ----------
        kernel_radius : numeric
            The radius (in meters) of the circular kernel used in the convolution.

        Returns
        -------
        convolution_result : ndarray

        """
        try:
            return self._convolution_result_dict[kernel_radius]
        except KeyError:
            # get circular kernel for the convolution
            kernel_pixel_radius = int(kernel_radius // self.res)
            kernel_pixel_len = 2 * kernel_pixel_radius + 1

            y, x = np.ogrid[
                -kernel_pixel_radius : kernel_pixel_len - kernel_pixel_radius,
                -kernel_pixel_radius : kernel_pixel_len - kernel_pixel_radius,
            ]
            mask = x * x + y * y <= kernel_pixel_radius * kernel_pixel_radius

            kernel = np.zeros(
                (kernel_pixel_len, kernel_pixel_len), dtype=self.lulc_dtype
            )
            kernel[mask] = 1

            # urban_mask = ndi.convolve(self.urban_lulc_arr, kernel)
            urban_mask = cv2.filter2D(
                self.urban_lulc_arr,
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_REFLECT,
            )

            # cache the convolution result
            self._convolution_result_dict[kernel_radius] = urban_mask

            # return it
            return urban_mask

    def compute_footprint_mask(
        self, kernel_radius, urban_threshold, num_patches=None, buffer_dist=None
    ):
        """Compute a boolean mask of the urban footprint of a given raster.

        Parameters
        ----------
        kernel_radius : numeric
            The radius (in meters) of the circular kernel used in the convolution.
        urban_threshold : float from 0 to 1
            Proportion of neighboring (within the kernel) urban pixels after which a
            given pixel is considered urban.
        num_patches : int, default 1
            The number of urban patches that should be featured in the returned
            urban/non-urban mask. If a value lower than one is provided, the returned
            urban/non-urban mask will feature all the urban patches. If no value is
            provided, the default value from `settings.DEFAULT_NUM_PATCHES` is used.
        buffer_dist : numeric, optional
            Distance to be buffered around the urban/non-urban mask. If no value is
            provided, no buffer is applied.

        Returns
        -------
        urban_mask : ndarray

        """
        kernel_pixel_radius = int(kernel_radius // self.res)
        kernel_pixel_len = 2 * kernel_pixel_radius + 1

        convolution_result = self.get_convolution_result(kernel_radius)

        urban_mask = convolution_result >= urban_threshold * (
            kernel_pixel_len * kernel_pixel_len - 1
        )

        if num_patches is not None and num_patches >= 1:
            # use the connected-component labelling to extract the largest contiguous
            # urban patches
            label_arr = ndi.label(urban_mask, KERNEL_MOORE)[0]
            # get the (pixel) counts of each urban patch label
            labels, counts = np.unique(label_arr, return_counts=True)
            # sort the urban patch label by (pixel) counts and delete the 0 (which in
            # `label_arr` always corresponds to the nodata values given the way
            # `ndi.label` works
            sorted_labels = labels[np.argsort(-counts)]
            sorted_labels = sorted_labels[sorted_labels > 0]
            # now let `urban_mask` include only the n-largest urban patches where n is
            # `num_patches`
            urban_mask = np.isin(label_arr, sorted_labels[:num_patches])

        if buffer_dist is not None:
            iterations = int(buffer_dist // self.res)
            urban_mask = ndi.binary_dilation(
                urban_mask, KERNEL_MOORE, iterations=iterations
            )

        return urban_mask.astype(self.mask_dtype)

    def compute_footprint_mask_shp(
        self,
        kernel_radius,
        urban_threshold,
        num_patches=1,
        buffer_dist=None,
        transform=None,
    ):
        """Compute a geometry of the urban footprint of a given raster.

        Parameters
        ----------
        kernel_radius : numeric
            The radius (in meters) of the circular kernel used in the convolution.
        urban_threshold : float from 0 to 1
            Proportion of neighboring (within the kernel) urban pixels after which a
            given pixel is considered urban.
        num_patches : int, default 1
            The number of urban patches that should be featured in the returned
            urban/non-urban mask. If `None` or a value lower than one is provided, the
            returned urban/non-urban mask will feature all the urban patches.
        buffer_dist : numeric, optional
            Distance to be buffered around the urban/non-urban mask. If no value is
            provided, no buffer is applied.
        transform : Affine, optional
            An affine transform matrix. Ignored if the instance was initialized with a
            path to a geotiff. If no transform is available, the geometry features will
            be generated based on pixel coordinates.

        Returns
        -------
        urban_mask_geom : GeometryCollection

        """
        urban_mask = self.compute_footprint_mask(
            kernel_radius,
            urban_threshold,
            num_patches=num_patches,
            buffer_dist=buffer_dist,
        )

        shapes_kws = {}
        if hasattr(self, "transform"):
            transform = self.transform
        if transform is not None:
            shapes_kws["transform"] = transform

        return geometry.GeometryCollection(
            [
                geometry.shape(geom)
                for geom, val in features.shapes(
                    urban_mask, mask=urban_mask, connectivity=8, **shapes_kws
                )
            ]
        )


def urban_footprint_mask(
    raster,
    kernel_radius,
    urban_threshold,
    urban_classes=None,
    num_patches=1,
    buffer_dist=None,
    res=None,
):
    """Compute a boolean mask of the urban footprint of a given raster.

    Parameters
    ----------
    raster : ndarray or str, file object or pathlib.Path object
        Land use/land cover (LULC) raster. If passing a ndarray (instead of the path to
        a geotiff), the resolution (in meters) must be passed to the `res` keyword
        argument.
    kernel_radius : numeric
        The radius (in meters) of the circular kernel used in the convolution.
    urban_threshold : float from 0 to 1
        Proportion of neighboring (within the kernel) urban pixels after which a given
        pixel is considered urban.
    urban_classes : int or list-like, optional
        Code or codes of the LULC classes that must be considered urban. Not needed if
        `raster` is already a boolean array of urban/non-urban LULC classes.
    num_patches : int, default 1
        The number of urban patches that should be featured in the returned
        urban/non-urban mask. If `None` or a value lower than one is provided, the
        returned urban/non-urban mask will feature all the urban patches.
    buffer_dist : numeric, optional
        Distance to be buffered around the urban/non-urban mask. If no value is
        provided, no buffer is applied.
    res : numeric, optional
        Resolution of the `raster` (assumes square pixels). Ignored if `raster` is a
        path to a geotiff.

    Returns
    -------
    urban_mask : ndarray

    """
    return UrbanFootprinter(raster, urban_classes, res=res).compute_footprint_mask(
        kernel_radius, urban_threshold, num_patches=num_patches, buffer_dist=buffer_dist
    )


def urban_footprint_mask_shp(
    raster,
    kernel_radius,
    urban_threshold,
    urban_classes=None,
    num_patches=1,
    buffer_dist=None,
):
    """Compute a geometry of the urban footprint of a given raster.

    Parameters
    ----------
    raster : str, file object or pathlib.Path object
        Path to the land use/land cover (LULC) raster.
    kernel_radius : numeric
        The radius (in meters) of the circular kernel used in the convolution.
    urban_threshold : float from 0 to 1
        Proportion of neighboring (within the kernel) urban pixels after which a given
        pixel is considered urban.
    urban_classes : int or list-like, optional
        Code or codes of the LULC classes that must be considered urban. Not needed if
        `raster` is already a boolean array of urban/non-urban LULC classes.
    num_patches : int, default 1
        The number of urban patches that should be featured in the returned
        urban/non-urban mask. If `None` or a value lower than one is provided, the
        returned urban/non-urban mask will feature all the urban patches.
    buffer_dist : numeric, optional
        Distance to be buffered around the urban/non-urban mask. If no value is
        provided, no buffer is applied.

    Returns
    -------
    urban_mask_geom : geometry

    """
    return UrbanFootprinter(raster, urban_classes).compute_footprint_mask_shp(
        kernel_radius, urban_threshold, num_patches=num_patches, buffer_dist=buffer_dist
    )
