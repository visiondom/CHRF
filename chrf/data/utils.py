import logging

import cv2
import numpy as np
import torch
from benchmark.utils.file_io import PathManager
from PIL import Image

from . import transforms as T

def convert_PIL_to_numpy(image, format=None):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "LAB"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "LAB":
        image = image.astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    return image


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray): an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        return convert_PIL_to_numpy(image, format)


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.

    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    else:
        if format == "L":
            image = image[:, :, 0]
        elif format == "LAB":
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB) * 255.0
            format = "RGB"
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image


def sampling_v2(frame_count, interval=10, num_key_frames=1):
    # itv = np.random.randint(1, interval)
    itv = np.random.choice([2, 5, interval], p=[0.4, 0.4, 0.2])
    start_idx = np.random.randint(0, max(1, frame_count - num_key_frames * itv))

    frame_idxs = [
        min(frame_count - 1, max(0, min(frame_count, start_idx + itv * i)))
        for i in range(num_key_frames + 1)
    ]

    return frame_idxs


def build_augmentation_ref(cfg, is_train=True):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    logger = logging.getLogger(__name__)

    angle = cfg.INPUT.ROTATE_ANGLE
    full_size = cfg.INPUT.FULL_SIZE

    tfm_gens = list()
    if is_train:
        tfm_gens.append(T.RandomRotation(angle))
    tfm_gens.append(T.Resize(full_size))

    if is_train:
        logger.info("Training augmentations for reference image use {}".format(tfm_gens))

    return tfm_gens


def build_augmentation_target(cfg, is_train=True):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    logger = logging.getLogger(__name__)

    full_size = cfg.INPUT.FULL_SIZE

    tfm_gens = []
    tfm_gens.append(T.Resize(full_size))

    if is_train:
        logger.info("Training augmentations for target image use {}".format(tfm_gens))

    return tfm_gens
