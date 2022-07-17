import os
from os.path import join as opj

import numpy as np

hier_map = {0: "species", 1: "genus", 2: "subfamily", 3: "family"}

def load_butterfly_data_multih(dataset_dir):
    img_dir = opj(dataset_dir, "images_small")

    cats = dict(
        species = _load_cates(opj(dataset_dir, "species.txt")),
        genus = _load_cates(opj(dataset_dir, "genus.txt")),
        subfamily = _load_cates(opj(dataset_dir, "subfamily.txt")),
        family = _load_cates(opj(dataset_dir, "family.txt"))
    )

    train_set_info = _load_data_info(opj(dataset_dir, "Butterfly200_train_release.txt"))
    val_set_info = _load_data_info(opj(dataset_dir, "Butterfly200_val_release.txt"))
    trainval_set_info = {}
    trainval_set_info.update(train_set_info)
    trainval_set_info.update(val_set_info)
    test_set_info = _load_data_info(opj(dataset_dir, "Butterfly200_test_release.txt"))

    trainval_set_dict = []
    for img, hier_labels in trainval_set_info.items():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
        }
        h_labels = {h_name: hier_labels[hier] for hier, h_name in hier_map.items()}
        category = {h_name: cats[h_name][label] for h_name, label in h_labels.items()}
        img_dict.update(h_labels)
        img_dict["category"] = category
        trainval_set_dict.append(img_dict)

    train_set_dict = []
    for img, hier_labels in train_set_info.items():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
        }
        h_labels = {h_name: hier_labels[hier] for hier, h_name in hier_map.items()}
        category = {h_name: cats[h_name][label] for h_name, label in h_labels.items()}
        img_dict.update(h_labels)
        img_dict["category"] = category
        train_set_dict.append(img_dict)

    val_set_dict = []
    for img, hier_labels in val_set_info.items():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
        }
        h_labels = {h_name: hier_labels[hier] for hier, h_name in hier_map.items()}
        category = {h_name: cats[h_name][label] for h_name, label in h_labels.items()}
        img_dict.update(h_labels)
        img_dict["category"] = category
        val_set_dict.append(img_dict)

    test_set_dict = []
    for img, hier_labels in test_set_info.items():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
        }
        h_labels = {h_name: hier_labels[hier] for hier, h_name in hier_map.items()}
        category = {h_name: cats[h_name][label] for h_name, label in h_labels.items()}
        img_dict.update(h_labels)
        img_dict["category"] = category
        test_set_dict.append(img_dict)

    return trainval_set_dict, train_set_dict, val_set_dict, test_set_dict

def _load_data_info(datapath):
    imgpath_labels_dict = {}
    with open(datapath) as f:
        lines = f.readlines()
    for line in lines:
        img_path, h0, h1, h2, h3 = line.strip().split(" ")
        imgpath_labels_dict[img_path] = [
            int(h0) - 1,
            int(h1) - 1,
            int(h2) - 1,
            int(h3) - 1,
        ]
    return imgpath_labels_dict

def _load_cates(classes_text):
    cates = {}
    with open(classes_text, "r", encoding='utf-8') as f:
        for line in f.readlines():
            cls_id, cate_name = line.strip().split(" ")
            cates[int(cls_id) - 1] = cate_name
    return cates
