import os
from os.path import join as opj

import numpy as np

hier_map = {0: "subclass", 1: "superclass"} # sup: 25 upper-level categories and sub: 292 subordinate classes

def load_vegfru_data_multih(dataset_dir):
    img_dir = opj(dataset_dir, "images")

    train_sup_info = _load_data_info(opj(dataset_dir, "vegfru_list", "supvegfru_train.txt"))
    val_sup_info = _load_data_info(opj(dataset_dir, "vegfru_list", "supvegfru_val.txt"))
    test_sup_info = _load_data_info(opj(dataset_dir, "vegfru_list", "supvegfru_test.txt"))

    train_sub_info = _load_data_info(opj(dataset_dir, "vegfru_list", "vegfru_train.txt"))
    val_sub_info = _load_data_info(opj(dataset_dir, "vegfru_list", "vegfru_val.txt"))
    test_sub_info = _load_data_info(opj(dataset_dir, "vegfru_list", "vegfru_test.txt"))

    trainval_sup_info = {}
    trainval_sup_info.update(train_sup_info)
    trainval_sup_info.update(val_sup_info)

    trainval_sub_info = {}
    trainval_sub_info.update(train_sub_info)
    trainval_sub_info.update(val_sub_info)

    trainval_set_dict = []
    for img in trainval_sub_info.keys():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
            "subclass": trainval_sub_info[img],
            "superclass": trainval_sup_info[img],
        }
        category = img.split('/')[-2]
        img_dict["category"] = category
        trainval_set_dict.append(img_dict)

    train_set_dict = []
    for img in train_sub_info.keys():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
            "subclass": train_sub_info[img],
            "superclass": train_sup_info[img],
        }
        category = img.split('/')[-2]
        img_dict["category"] = category
        train_set_dict.append(img_dict)

    val_set_dict = []
    for img in val_sub_info.keys():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
            "subclass": val_sub_info[img],
            "superclass": val_sup_info[img],
        }
        category = img.split('/')[-2]
        img_dict["category"] = category
        val_set_dict.append(img_dict)

    test_set_dict = []
    for img in test_sub_info.keys():
        img_path = img
        img_dict = {
            "img": opj(img_dir, img_path),
            "subclass": test_sub_info[img],
            "superclass": test_sup_info[img]
        }
        category = img.split('/')[-2]
        img_dict["category"] = category
        test_set_dict.append(img_dict)

    return trainval_set_dict, train_set_dict, val_set_dict, test_set_dict

def _load_data_info(datapath):
    imgpath_labels_dict = {}
    with open(datapath) as f:
        lines = f.readlines()
    for line in lines:
        img_path, h0 = line.strip().split(" ")
        imgpath_labels_dict[img_path] = int(h0)
    return imgpath_labels_dict

def _load_cates(classes_text):
    cates = {}
    with open(classes_text, "r", encoding='utf-8') as f:
        for line in f.readlines():
            cls_id, cate_name = line.strip().split(" ")
            cates[int(cls_id) - 1] = cate_name
    return cates
