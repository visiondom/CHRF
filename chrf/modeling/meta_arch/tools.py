import os
from typing import Optional

import torch
from chrf.data.utils import convert_image_to_rgb
from chrf.utils.visualizer import Visualizer


def state(tensor: torch.Tensor):
    print(tensor.mean(), tensor.std(), tensor.max(), tensor.mean())


def draw_patch(img, coords, coords_shape, stride, bbox):
    vis_patch_ref = Visualizer(img)
    bbox_i = bbox * stride + stride / 2
    l, r, t, b = bbox_i
    bbox_i = (l, t, r, b)
    vis_patch_ref.draw_box(bbox_i, edge_color="red")

    coords_i = coords * stride + stride / 2
    coords_i = coords_i.detach().cpu().numpy()
    vis_patch_ref.draw_coordinates(coords_i, grid_shape=coords_shape)

    vis_patch_img = vis_patch_ref.get_output().get_image()
    return vis_patch_img


def vis_batch(
    batched_inputs, results, input_format: Optional[str] = None, save_dir: Optional[str] = None
):
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    stride = results["stride"]
    affinity = results["affinity"]
    grid = results["grid"]
    patch_grid = results["patch_grid"]
    grid_shape = results["grid_shape"]
    coordinates = results["coordinates"]
    bbox = results["bbox"]

    for idx, input in enumerate(batched_inputs):
        tgt_img = input["target"]
        tgt_img = convert_image_to_rgb(tgt_img.permute(1, 2, 0), input_format)
        ref_img = input["reference"]
        ref_img = convert_image_to_rgb(ref_img.permute(1, 2, 0), input_format)

        patch_box = input["patch_box"]
        x, y, w, h = patch_box.numpy().tolist()
        patch_box = (x, y, x + w, y + h)
        patch_grid_i = patch_grid[idx] * stride
        patch_grid_i[:, 0] += x + stride / 2
        patch_grid_i[:, 1] += y + stride / 2

        bbox_i = bbox[idx] * stride
        l, r, t, b = bbox_i
        bbox_i = (l, t, r, b)

        coordinates_i = coordinates[idx] * stride

        vis_ref = Visualizer(ref_img)
        vis_target = Visualizer(tgt_img)
        vis_ref.draw_box(patch_box)
        vis_target.draw_box(bbox_i)
        vis_ref.draw_coordinates(patch_grid_i, grid_shape=grid_shape)
        vis_target.draw_coordinates(coordinates_i, grid_shape=grid_shape)

        if save_dir is not None:
            sub_dir = os.path.join(save_dir, "{}".format(idx))
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            vis_ref.get_output().save(os.path.join(sub_dir, "ref.png"))
            vis_target.get_output().save(os.path.join(sub_dir, "tar.png"))
