import colorsys
import os
from enum import Enum, unique

import cv2
import imageio as io
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision.transforms.functional as tF
from chrf.data.utils import read_image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.decomposition import PCA

from .colormap import _DAVIS_COLORS, _VIP_CLASSES, _VIP_COLORS, GridColorMap, random_color

_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


def pca_feat(feat, K=1, solver="auto", whiten=True, norm=True):
    if isinstance(feat, torch.Tensor):
        feat = feat.cpu()

    N, C, H, W = feat.shape
    pca = PCA(
        n_components=3 * K,
        svd_solver=solver,
        whiten=whiten,
    )

    feat = feat.permute(0, 2, 3, 1)
    feat = feat.reshape(-1, C).numpy()

    pca_feat = pca.fit_transform(feat)
    pca_feat = torch.Tensor(pca_feat).view(N, H, W, K * 3)
    pca_feat = pca_feat.permute(0, 3, 1, 2)

    pca_feat = [pca_feat[:, k : k + 3] for k in range(0, pca_feat.shape[1], 3)]
    if norm:
        pca_feat = [(x - x.min()) / (x.max() - x.min()) for x in pca_feat]
        # rescale to [0-255] for visulization
        pca_feat = [x * 255.0 for x in pca_feat]

    return pca_feat[0] if K == 1 else pca_feat


def make_gif(video, save_path=None, size=256):
    if isinstance(video, str) and os.path.exists(video):
        image_names = sorted(os.listdir(video))
        images = [read_image(os.path.join(video, image_name)) for image_name in image_names]
        video = images

    assert isinstance(video, list or tuple)
    assert isinstance(video[0], np.ndarray)

    video = [cv2.resize(image, (size, size)) for image in video]
    if save_path is None:
        return video

    io.mimsave(save_path, video, duration=0.2)


class Visualizer:
    def __init__(self, img_rgb, scale=1.0):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )

    def draw_coordinates(self, coordinates, grid_shape=(32, 32), radius=3, colors=None):
        """

        Args:
            coordinates:
            grid_shape:
            colors:

        Returns:

        """
        if colors is None:
            colors = ["blue", "orange", "red", "yellow"]
            colors = [mplc.to_rgba(_) for _ in colors]
            colors = np.array(colors).reshape((2, 2, 4))

        color_map = GridColorMap(grid_shape, colors)
        for idx, coord in enumerate(coordinates):
            color = color_map(idx)
            self.draw_circle(coord, color=color, radius=radius)

    def draw_davis_seg(self, instance_seg, area_threshold=None, alpha=0.5):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(instance_seg, torch.Tensor):
            instance_seg = instance_seg.numpy()

        labels = np.unique(instance_seg)
        labels = sorted(labels.tolist())

        for i, label in enumerate(labels[1:]):
            binary_mask = (instance_seg == label).astype(np.uint8)
            text = "{}".format(label)
            self.draw_binary_mask(
                binary_mask,
                color=_DAVIS_COLORS[i],
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        return self.output

    def draw_vip_seg(self, instance_seg, area_threshold=None, alpha=0.5):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(instance_seg, torch.Tensor):
            instance_seg = instance_seg.numpy()

        labels, _ = np.unique(instance_seg, return_counts=True)

        for i, label in enumerate(labels[1:]):
            binary_mask = (instance_seg == label).astype(np.uint8)
            text = "{}".format(_VIP_CLASSES[label])
            self.draw_binary_mask(
                binary_mask,
                color=_VIP_COLORS[label],
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        return self.output

    """
        Primitive drawing functions:
    """

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_color=None,
        text=None,
        alpha=0.5,
        area_threshold=0
    ):
        """
        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn in the object's center of mass.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component small than this will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba)

        if text is not None and has_valid_segment:
            # TODO sometimes drawn on wrong objects. the heuristics here can improve.
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, 8
            )
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            # draw text on the largest component, as well as other very large components.
            for cid in range(1, _num_cc):
                if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                    # median is more stable than centroid
                    # center = centroids[largest_component_id]
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                    self.draw_text(text, center, color=lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    @classmethod
    def _change_color_brightness(cls, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(
            polygon_color[0], modified_lightness, polygon_color[2]
        )
        return modified_color

    def get_output(self):
        """
        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output


def grid_paste(grid_patches, margin=2, background=(0, 0, 0)):
    """
    Paste patches to be looked similar to the original image for visulization.
    Args:
        grid_patches: B x H x W x c x h x w Tesnsor
        margin: margin between patches
        background: background RGB color
    Return:
        background with patches pasted in grid fashion
    """
    if grid_patches.is_cuda:
        grid_patches = grid_patches.detach().cpu()
    B, H, W, c, h, w = grid_patches.shape
    background_corlor = torch.tensor(background, dtype=torch.float32).reshape(
        (1, 1, 1, c, 1, 1)
    )
    top_margin = torch.zeros(*(B, H, W, c, margin, w + margin)) + background_corlor
    left_margin = torch.zeros(*(B, H, W, c, h, margin)) + background_corlor
    grid_patches_with_margin = torch.cat((left_margin, grid_patches), dim=-1)
    grid_patches_with_margin = torch.cat(
        (top_margin, grid_patches_with_margin), dim=-2
    )  # B x H x W x c x h_m x w_m
    pasted_patches = (
        grid_patches_with_margin.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .flatten(-2, -1)
        .flatten(-3, -2)
    )  # B x c x (H*h_m) x (W*w_m)
    B, c, h_p, w_p = pasted_patches.shape
    full_bg_color = torch.tensor(background, dtype=torch.float32).reshape((1, c, 1, 1))
    right_padding = torch.zeros(*(B, c, h_p, margin)) + full_bg_color
    pasted_patches = torch.cat((pasted_patches, right_padding), dim=-1)
    bottom_padding = torch.zeros(*(B, c, margin, w_p + margin)) + full_bg_color
    pasted_patches = torch.cat((pasted_patches, bottom_padding), dim=-2)
    return pasted_patches


def rearrange_affinity(batch_affinity, rearrange, zoom_scale=3):
    """
    Zoom , rearrange ang turning to heatmap for a batch of affinities to make elements more obvious.
    Args:
        batch_affinity: B x N1 x N2 Affinity tensor softmaxed along rows.
        rearrange: do nothing if None or given (h1, w1, h2, w2) to rearrange affinities in grid fashion.
        zoom_scale (int): scale factor for affinity elements
    Return:
        modified batch of affinities.
    """
    assert isinstance(zoom_scale, int) and zoom_scale > 0, "Invalid zoom scale"
    if batch_affinity.is_cuda:
        batch_affinity = batch_affinity.detach().cpu()
    B, N1, N2 = batch_affinity.shape
    h1, w1, h2, w2 = rearrange
    assert h1 * w1 == N1 and h2 * w2 == N2, "Invalid rearrange parameter !"
    rearrange_batch_affinity = batch_affinity.reshape((B, h1, w1, h2, w2)).contiguous()
    # TODO more efficiently heatmap converting
    rearrange_batch_heat = torch.zeros(*(B, h1, w1, 3, h2, w2))
    for b in range(B):
        for h in range(h1):
            for w in range(w1):
                patch = rearrange_batch_affinity[b, h, w].numpy()
                heatpatch = cv2.applyColorMap(np.uint8(255 * patch), cv2.COLORMAP_JET)
                heatpatch = cv2.cvtColor(heatpatch, cv2.COLOR_BGR2RGB)
                rearrange_batch_heat[b, h, w] = (
                    torch.tensor(heatpatch, dtype=torch.float32).permute(-1, 0, 1) / 255
                )
    # and grid line
    grid_rearrange_batch_heat = grid_paste(
        rearrange_batch_heat, margin=1, background=(1, 1, 1)
    )  # B x 3 x (h1*h3) x (w1*w3)
    batch_affinity = grid_rearrange_batch_heat.unsqueeze(-1).unsqueeze(-1)
    expand_batch_affinity = batch_affinity.repeat(
        1, 1, 1, 1, zoom_scale, zoom_scale
    )  # B x 3 x (h1*h3) x (w1*w3) x s x s
    expand_batch_affinity = (
        expand_batch_affinity.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .flatten(-2, -1)
        .flatten(-3, -2)
    )  # B x 3 x (h1*h3*s) x (w1*w3*s)
    return expand_batch_affinity


def aff2heat(aff_batch):
    aff_batch = aff_batch.detach().cpu()
    if aff_batch.dim() == 2:
        aff_batch = aff_batch.unsqueeze(0)
    heats = []
    for b in range(aff_batch.shape[0]):
        aff_arr = aff_batch[b].numpy()
        heat_arr = cv2.applyColorMap(np.uint8(255 * aff_arr), cv2.COLORMAP_JET)
        heat_arr = cv2.cvtColor(heat_arr, cv2.COLOR_BGR2RGB)
        heats.append(heat_arr)
    return heats


def _cv_match(queryIdx, trainIdx, distance=0):
    match = cv2.DMatch()
    match.distance = distance
    match.queryIdx = queryIdx
    match.trainIdx = trainIdx
    match.imgIdx = 0
    return match


def _cv_point(pt, orientation=0):
    point = cv2.KeyPoint()
    point.size = 17
    point.angle = orientation
    point.class_id = -1
    point.octave = 0
    point.response = 0
    point.pt = (pt[0], pt[1])
    return point


def draw_corrs(img1_rgb, img2_rgb, coords1, coords2, tensor_result=True):
    num_coords = coords1.shape[0]
    img1_points = []
    img2_points = []
    matches = []
    for i in range(num_coords):
        img1_points.append(_cv_point(coords1[i]))
        img2_points.append(_cv_point(coords2[i]))
        matches.append(_cv_match(i, i))
    result = np.empty(
        (max(img1_rgb.shape[0], img2_rgb.shape[0]), img1_rgb.shape[1] + img2_rgb.shape[1], 3),
        dtype=np.uint8,
    )
    cv2.drawMatches(
        img1_rgb,
        img1_points,
        img2_rgb,
        img2_points,
        matches,
        result,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if tensor_result:
        result = tF.to_tensor(result)
    return result
