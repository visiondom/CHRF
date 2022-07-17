import os
from os.path import join as opj

import numpy as np

hier_map = {0: "class", 1: "genus", 2: "family", 3: "order"}

def load_cub_data_multih(dataset_dir):
    img_dir = opj(dataset_dir, "images")
    gaze_img_dir = {}
    gaze_img_dir["class"] = opj(dataset_dir, "gmaps")
    for hier, h_name in hier_map.items():
        gaze_img_dir[h_name] = opj(dataset_dir, "gmaps_{}".format(hier_map[hier] if hier != 0 else 'species'))
    boxes = _load_boxes(opj(dataset_dir, "bounding_boxes.txt"))
    cates = _load_cates(opj(dataset_dir, "classes.txt"))
    cls_labels = _load_img_cls_labels(opj(dataset_dir, "image_class_labels.txt"))
    hier_labels = _load_hier_labels(dataset_dir)
    train_test_split = _load_split(opj(dataset_dir, "train_test_split.txt"))
    train_set_dict = []
    test_set_dict = []
    with open(opj(dataset_dir, "images.txt"), "r") as f:
        for line in f.readlines():
            img_id_str, img_path = line.strip().split(" ")
            img_idx = int(img_id_str) - 1
            gmaps_paths = {}
            for h_name in gaze_img_dir:
                gmaps_paths[h_name] = _parse_gaze(img_path, gaze_img_dir[h_name])


            no_gmap = False
            for hier, gmaps_path in gmaps_paths.items():
                for gpath in gmaps_path.values():
                    if not os.path.exists(gpath):
                        no_gmap = True
                        break
                if no_gmap:
                    break
            if no_gmap:
                continue


            h_labels = {
                h_name: hier_labels[img_path][hier] for hier, h_name in hier_map.items()
            }
            img_dict = {
                "img": opj(img_dir, img_path),
                "obj_box": boxes[img_idx],
                "category": cates[cls_labels[img_idx]],
                "gmaps": gmaps_paths,
            }
            img_dict.update(h_labels)
            if train_test_split[img_idx]:
                train_set_dict.append(img_dict)
            else:
                test_set_dict.append(img_dict)
    return train_set_dict, test_set_dict


def _load_hier_labels(datadir):
    imgpath_labels_dict = {}
    with open(opj(datadir, "train_images_4_level_V1.txt")) as f:
        lines = f.readlines()
    with open(opj(datadir, "test_images_4_level_V1.txt")) as f:
        lines.extend(f.readlines())
    for line in lines:
        img_path, h0, h1, h2, h3 = line.strip().split(" ")
        imgpath_labels_dict[img_path] = [
            int(h0) - 1,
            int(h1) - 1,
            int(h2) - 1,
            int(h3) - 1,
        ]
    return imgpath_labels_dict


def _load_boxes(box_text):
    boxes = {}
    with open(box_text, "r") as f:
        for line in f.readlines():
            arr = np.array(line.strip().split(" "), dtype=np.float32)
            boxes[int(arr[0]) - 1] = arr[1:]
    return boxes


def _load_cates(classes_text):
    cates = {}
    with open(classes_text, "r") as f:
        for line in f.readlines():
            cls_id, cate_name = line.strip().split(" ")
            cates[int(cls_id) - 1] = cate_name
    return cates


def _load_img_cls_labels(label_text):
    labels = {}
    with open(label_text, "r") as f:
        for line in f.readlines():
            img_id, cls_id = line.strip().split(" ")
            labels[int(img_id) - 1] = int(cls_id) - 1
    return labels


def _load_split(split_text):
    splits = {}
    with open(split_text, "r") as f:
        for line in f.readlines():
            img_id, train_flag = line.strip().split(" ")
            splits[int(img_id) - 1] = int(train_flag) == 1
    return splits


def _parse_gaze(sub_img_path, gmap_dir):
    """
    return:
        {
            "gaze_coarse": map.png path,
            "gaze_mid": mapmg.png path,
            "gaze_fg": mapfg.png path,
            ...
        }
    """
    subdir, img_fname = os.path.split(sub_img_path)
    img_fid = img_fname.split(".")[0]
    return {
        "gaze_coarse": opj(gmap_dir, img_fid, img_fid + "-map.png"),
        "gaze_mid": opj(gmap_dir, img_fid, img_fid + "-mapmg.png"),
        "gaze_fg": opj(gmap_dir, img_fid, img_fid + "-mapfg.png"),
    }
