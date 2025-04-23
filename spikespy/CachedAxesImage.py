import matplotlib.transforms as mtransforms
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import AxesImage
import numpy as np
class CachedAxesImage(AxesImage):
    """
    A subclass of AxesImage that caches the rendered image and updates
    only when the x or y limits of the axes or the figure size changes.
    """

    def __init__(self, ax, **kwargs):
        super().__init__(ax, **kwargs)
        self._cached_image = None
        self._cached_extent = None
        self._cached_figure_size = None
        self._cached_lims = None
        self._cached_clim = None
        self._cached_clip = None
        self._cached_transform = None


    
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        current_extent = self.get_extent()
        current_figure_size = self.axes.figure.get_size_inches()
        current_xlim = self.axes.get_xlim()
        current_ylim = self.axes.get_ylim()
        current_clim = self.get_clim()

        trans = self.get_transform()
        x1, x2, y1, y2 = self.get_extent()
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        transformed_bbox = TransformedBbox(bbox, trans)
        # the transformedbbox is the bounding box of the image in display coordinates


        clip = ((self.get_clip_box() or self.axes.bbox) if self.get_clip_on()
                else self.get_figure(root=True).bbox)

            

        # Check if the cached image is valid
        if (
            self._cached_image is not None
            and self._cached_extent == current_extent
            and self._cached_figure_size == tuple(current_figure_size)
            and self._cached_magnification == magnification
            and self._cached_lims == (current_xlim, current_ylim)
            # check ratio of xlim and ylim
            and np.isclose((abs((current_xlim[1] - current_xlim[0]) / (current_ylim[1] - current_ylim[0]))), abs((self._cached_lims[0][1] - self._cached_lims[0][0]) / (self._cached_lims[1][1] - self._cached_lims[1][0])))
            and self._cached_clim == current_clim
            
            #and (self._cached_clip is not None and self._cached_clip.fully_overlaps(Bbox.intersection(transformed_bbox, clip)))
            
            and not self.stale
        ):
            return self._cached_image
            # write the transform that gets from self._cached_clip to clip
            # clip2 = Bbox.intersection(transformed_bbox, self._cached_clip)
            # x0 = clip2.x0  #transformed_bbox.xmin-self._cached_clip.xmin 
            # y0 = clip2.y0#transformed_bbox.ymin-self._cached_clip.ymin 

            
            # print(f"transformed_bbox:{transformed_bbox.extents}; cached_clip:{self._cached_clip.extents}; clip2:{clip2.extents}; x0:{x0}; y0:{y0}")
            # return self._cached_image[0],x0,y0, None# self._cached_image[3]

        self._cached_clip = clip #clip.expanded(1.2, 1.2)
        # Cache the new larger cut of the data
        fig = self.get_figure(root=True)
        fac = renderer.dpi/fig.dpi
        self._cached_image = self._make_image(
            self._A, bbox, transformed_bbox, self._cached_clip,
             magnification=magnification / fac, unsampled=unsampled
        )
        
        clip2 = Bbox.intersection(transformed_bbox, self._cached_clip)
        self._cached_extent = current_extent
        self._cached_figure_size = tuple(current_figure_size)
        self._cached_magnification = magnification
        self._cached_lims = (current_xlim, current_ylim)
        self._cached_clim = current_clim
        #self.set_clip_on(False)

        return self._cached_image#[0],clip2.x0,clip2.y0, None#self._cached_image[3]