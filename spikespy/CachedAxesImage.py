import matplotlib.transforms as mtransforms
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import AxesImage
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
class CachedAxesImage(AxesImage):
    """
    A subclass of AxesImage that caches the rendered image and updates
    only when the x or y limits of the axes or the figure size changes.
    """

    def __init__(self, ax, rolling_max=False,  **kwargs):
        super().__init__(ax, **kwargs)
        self._cached_image = None
        self._cached_extent = None
        self._cached_figure_size = None
        self._cached_lims = None
        self._cached_clim = None
        self._cached_clip = None
        self._cached_transform = None
        self.rolling_max= rolling_max


    
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        current_extent = self.get_extent()
        current_figure_size = self.axes.figure.get_size_inches()
        current_xlim = self.axes.get_xlim()
        current_ylim = self.axes.get_ylim()
        current_clim = self.get_clim()

        clip = ((self.get_clip_box() or self.axes.bbox) if self.get_clip_on()
                else self.get_figure(root=True).bbox)
        trans = self.get_transform()
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        # the transformedbbox is the bounding box of the image in display coordinates
        clipped_transformed_bbox =Bbox.intersection(transformed_bbox, clip)
        if clipped_transformed_bbox is None:
            return None,0,0,None
        clipped_bbox = clipped_transformed_bbox.transformed(trans.inverted())
        clip2 = None
        if self._cached_clip is not None:
            clip2 = self._cached_clip.transformed(trans)
            

        # Check if the cached image is valid
        if (
            self._cached_image is not None
            and self._cached_extent == current_extent
            and self._cached_figure_size == tuple(current_figure_size)
            and self._cached_magnification == magnification
            # and self._cached_lims == (current_xlim, current_ylim)
            # check ratio of xlim and ylim
            and np.isclose((abs((current_xlim[1] - current_xlim[0]) / (current_ylim[1] - current_ylim[0]))), abs((self._cached_lims[0][1] - self._cached_lims[0][0]) / (self._cached_lims[1][1] - self._cached_lims[1][0])))
            and self._cached_clim == current_clim
            
            and (self._cached_clip is not None and max(self._cached_clip.xmin,0) <= max(clipped_bbox.xmin,0) and min(self._cached_clip.xmax,self._A.shape[0]) >= min(clipped_bbox.xmax,self._A.shape[0]) and max(self._cached_clip.ymin,0) <= max(clipped_bbox.ymin,0) and min(self._cached_clip.ymax,self._A.shape[1]) >= min(clipped_bbox.ymax, self._A.shape[1]))
            
            and not self.stale
        ):
            # return self._cached_image
            # write the transform that gets from self._cached_clip to clip
            clip2 = Bbox.intersection(transformed_bbox, clip2)
            x0 = clip2.x0# self._cached_clip.xmin- transformed_bbox.xmin
            y0 = clip2.y0 #self._cached_clip.ymin- transformed_bbox.ymin
            return self._cached_image[0],x0,y0, None# self._cached_image[3]

        clip2 = clip.expanded(2, 2) # 2x the width and height of the window.
        # Cache the new larger cut of the data
        fig = self.get_figure()
        fac = renderer.dpi/fig.dpi
        # hack for ax if there are too many points is too big do a rolling max interpolation on X axis
        A = self._A.copy()
        if self.rolling_max:
            x_scale = (bbox.width/transformed_bbox.width)/2
            if x_scale > 1:
                

                # rolling window on X axis of x_scale points
                
                pad_width = int(x_scale) - 1
                A = np.pad(A, ((0, 0), (0, pad_width)), mode='edge')
                A = sliding_window_view(A, int(x_scale), axis=1)
                A = np.nanmax(A, axis=2)

            y_scale = (bbox.height/transformed_bbox.height)/2
            if y_scale > 1:
                # rolling window on Y axis of y_scale points
                pad_width = int(y_scale) - 1
                A = np.pad(A, ((0, pad_width), (0, 0)), mode='edge')
                A = sliding_window_view(A, int(y_scale), axis=0)
                A = np.nanmax(A, axis=2)

        self._cached_image = self._make_image(
            A, bbox, transformed_bbox, clip2,
             magnification=magnification / fac, unsampled=unsampled
        )
        
        self._cached_clip = Bbox.intersection(transformed_bbox, clip2).transformed(trans.inverted())
        self._cached_extent = current_extent
        self._cached_figure_size = tuple(current_figure_size)
        self._cached_magnification = magnification
        self._cached_lims = (current_xlim, current_ylim)
        self._cached_clim = current_clim
        print("update cached image")
        # print(f"ORIGINAL: transformed_bbox:{transformed_bbox.extents}; cached_clip:{self._cached_clip.extents};  x0:{self._cached_clip.x0}; y0:{self._cached_clip.y0}")
        # print(f"ORIGINAL: clip orignal : {self._cached_clip.transformed(trans.inverted()).extents};")

        return self._cached_image