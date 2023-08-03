"""Windowed urban footprinter."""
import cv2
import numpy as np
import rasterio as rio
from rasterio import windows

from urban_footprinter import UrbanFootprinter, settings


class WindowedUrbanFootprinter(UrbanFootprinter):
    """Windowed urban footprinter."""

    def __init__(
        self,
        raster,
        kernel_radius,
        window_width,
        window_height,
        init_center,
        urban_classes=None,
        convolve_border_type=None,
        lulc_dtype=None,
        mask_dtype=None,
    ):
        """Initialize the urban footprinter.

        Parameters
        ----------
        raster : str, file object or pathlib.Path object
            Land use/land cover (LULC) raster. The value can be a filename or URL, a
            file-like object opened in binary ('rb') mode, or a Path object that will be
            passed to `rasterio.open`.
        kernel_radius : numeric
            The radius (in meters) of the circular kernel used in the convolution.
        window_width, window_height : int
            Window width and height (in number of pixels).
        init_center : tuple of int or shapely.geometry.Point
            Initial center coordinates (x, y) of the window, either as a tuple of int
            representing the (row, col) coordinates of the center pixel, or as a shapely
            Point in the raster's coordinate reference system.
        urban_classes : int or list-like, optional
            Code or codes of the LULC classes that must be considered urban. Not needed
            if `raster` is already a boolean array of urban/non-urban LULC classes.
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
        # with rio.open(raster) as src:
        #     raster = src.read(1)
        #     res = src.res[0]  # only square pixels are supported
        #     self.transform = src.transform
        # self.res = res
        with rio.open(raster) as src:
            self.res = src.res[0]  # only square pixels are supported
        self.raster = raster

        # TODO: DRY this by creating an abstract base class for the windowed and
        # non-windowed footprinters
        # -- code to DRY starts here --
        if convolve_border_type is None:
            convolve_border_type = settings.DEFAULT_CONV_BORDER_TYPE
        self.convolve_border_type = convolve_border_type
        if lulc_dtype is None:
            lulc_dtype = settings.DEFAULT_LULC_DTYPE
        self.lulc_dtype = lulc_dtype
        if mask_dtype is None:
            mask_dtype = settings.DEFAULT_MASK_DTYPE
        self.mask_dtype = mask_dtype
        # -- code to DRY ends here --

        # get circular kernel for the convolution
        # we need to do this *after* setting `self.lulc_dtype`
        kernel_pixel_radius = int(kernel_radius // self.res)
        kernel_pixel_len = 2 * kernel_pixel_radius + 1

        y, x = np.ogrid[
            -kernel_pixel_radius : kernel_pixel_len - kernel_pixel_radius,
            -kernel_pixel_radius : kernel_pixel_len - kernel_pixel_radius,
        ]
        mask = x * x + y * y <= kernel_pixel_radius * kernel_pixel_radius

        kernel = np.zeros((kernel_pixel_len, kernel_pixel_len), dtype=self.lulc_dtype)
        kernel[mask] = 1
        # set central point to zero so that we do not count the target pixel itself
        kernel[kernel_pixel_radius, kernel_pixel_radius] = 0
        self.kernel = kernel

        # urban classes
        if urban_classes is None:
            # we assume that the raster is already a boolean array of urban/non-urban
            self.is_urban = lambda lulc_arr: lulc_arr
        else:
            self.is_urban = lambda lulc_arr: np.isin(lulc_arr, urban_classes)

        # process window attributes
        self.window_width = window_width
        self.window_height = window_height
        if isinstance(init_center, tuple):
            row, col = init_center
        else:  # assume shapely Point
            with rio.open(raster) as src:
                row, col = src.index(*init_center.coords[0])
        self.init_window = windows.Window(
            col - self.window_width // 2,
            row - self.window_height // 2,
            self.window_width,
            self.window_height,
        )
        # add the kernel radius to the window dimensions and transform the window
        # dimensions from meters to pixels
        # Note that this assumes square pixels
        self.pad_size = int(kernel_radius / self.res)

    @staticmethod
    def get_opposite_direction(direction):
        """Get the opposite direction."""
        if direction == "left":
            return "right"
        elif direction == "right":
            return "left"
        elif direction == "up":
            return "down"
        elif direction == "down":
            return "up"
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @staticmethod
    def get_neighbour_window(window, direction):
        """Get neighbour window."""
        if direction == "left":
            return windows.Window(
                window.col_off - window.width,
                window.row_off,
                window.width,
                window.height,
            )
        elif direction == "right":
            return windows.Window(
                window.col_off + window.width,
                window.row_off,
                window.width,
                window.height,
            )
        elif direction == "up":
            return windows.Window(
                window.col_off,
                window.row_off - window.height,
                window.width,
                window.height,
            )
        elif direction == "down":
            return windows.Window(
                window.col_off,
                window.row_off + window.height,
                window.width,
                window.height,
            )
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @staticmethod
    def get_arr_border(arr, direction):
        """Get array border."""
        if direction == "left":
            return arr[:, 0]
        elif direction == "right":
            return arr[:, -1]
        elif direction == "up":
            return arr[0, :]
        elif direction == "down":
            return arr[-1, :]
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @staticmethod
    def extend_mask(base_mask, window_mask, direction):
        """Extend the mask."""
        if direction == "left":
            return np.hstack((window_mask, base_mask[:, :-1]))
        elif direction == "right":
            return np.hstack((base_mask[:, 1:], window_mask))
        elif direction == "up":
            return np.vstack((window_mask, base_mask[:-1, :]))
        elif direction == "down":
            return np.vstack((base_mask[1:, :], window_mask))
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def pad_window(self, window):
        """Pad window."""
        return windows.Window(
            window.col_off - self.pad_size,
            window.row_off - self.pad_size,
            window.width + 2 * self.pad_size,
            window.height + 2 * self.pad_size,
        )

    def _compute_window_footprint_mask(
        self,
        window,
        urban_threshold,
        buffer_dist=None,
    ):
        with rio.open(self.raster) as src:
            lulc_arr = src.read(1, window=window)
            # `num_patches` has to be `None` so that all patches are returned
        convolution_result = cv2.filter2D(
            # note that the array must be of type `self.lulc_dtype`
            self.is_urban(lulc_arr).astype(self.lulc_dtype),
            ddepth=-1,
            kernel=self.kernel,
            borderType=cv2.BORDER_REFLECT,
        )
        # `convolution_result` is the number of surrounding pixels that are urban
        return convolution_result >= urban_threshold * self.kernel.sum()

    def compute_footprint_mask(self, urban_threshold, buffer_dist=None):
        """Compute a boolean mask of the urban footprint of a given raster.

        Parameters
        ----------
        urban_threshold : float from 0 to 1
            Proportion of neighboring (within the kernel) urban pixels after which a
            given pixel is considered urban.
        buffer_dist : numeric, optional
            Distance to be buffered around the urban/non-urban mask. If no value is
            provided, no buffer is applied.

        Returns
        -------
        urban_mask : ndarray
        """
        window = self.init_window
        urban_mask = self._compute_window_footprint_mask(
            self.pad_window(window),
            urban_threshold,
            buffer_dist=buffer_dist,
        )
        # idea: get the largest urban patch of the initial window and check whether
        # there are urban pixels in each of the 4 directions. If there are, then check
        # the corresponding direction separately.
        # ACHTUNG: this assumes that the largest patch of the initial window is the
        # target patch and discards other patches that may be larger
        neighbour_mask_dict = {}
        for direction in ["left", "right", "up", "down"]:
            if np.any(self.get_arr_border(urban_mask, direction)):
                neighbour_window = self.get_neighbour_window(window, direction)
                neighbour_mask = self._compute_window_footprint_mask(
                    self.pad_window(neighbour_window),
                    urban_threshold,
                    buffer_dist=buffer_dist,
                )
                # add the neighbour mask to the urban mask depending on the direction
                neighbour_mask_dict[direction] = neighbour_mask

        return urban_mask, neighbour_mask_dict
