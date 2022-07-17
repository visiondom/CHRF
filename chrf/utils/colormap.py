import numpy as np

import math

# fmt: off
# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on
_DAVIS_COLORS = [_COLORS[i] for i in range(10)]


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


class GridColorMap:
    def __init__(self, grid_shape, colors: np.ndarray):
        assert colors.ndim == 3

        self.grid_shape = grid_shape
        self.colors = colors

    def __call__(self, idx):
        g_h, g_w = self.grid_shape
        c_h, c_w = self.colors.shape[:2]

        g_h_per_color = math.ceil(g_h / c_h)
        g_w_per_color = math.ceil(g_w / c_w)

        h, w = idx // g_w, idx % g_w
        c_idx_h = h // g_h_per_color
        c_idx_w = w // g_w_per_color

        color = self.colors[c_idx_h, c_idx_w]

        return color


def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    index_map = {}
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(
                r,
                np.left_shift(
                    np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7 - j
                ),
            )
            g = np.bitwise_or(
                g,
                np.left_shift(
                    np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7 - j
                ),
            )
            b = np.bitwise_or(
                b,
                np.left_shift(
                    np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7 - j
                ),
            )
            cid = np.right_shift(cid, 3)

        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
        index_map["%d_%d_%d" % (r, g, b)] = i
    return color_map, index_map


_VIP_CLASSES = [
    "background",
    "hat",
    "hair",
    "sun-glasses",
    "upper-clothes",
    "dress",
    "coat",
    "socks",
    "pants",
    "gloves",
    "scarf",
    "skirt",
    "torso-skin",
    "face",
    "right-arm",
    "left-arm",
    "right-leg",
    "left-leg",
    "right-shoe",
    "left-shoe",
]
# _VIP_COLORS = _get_voc_color_map()[0] / 255
_VIP_COLORS = [_COLORS[i] for i in range(20)]


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)
