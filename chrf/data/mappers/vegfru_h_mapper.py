import copy
import os
import chrf.data.transforms as dF
import cv2
import pickle
import numpy as np
import torch
import torchvision.transforms as tT
import torchvision.transforms.functional as tF

class VegFruClassifierMapper(object):
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augs = (
            dF.AugmentationList(
                [
                    dF.Resize(cfg.INPUT.FULL_SIZE),
                    dF.RandomCrop(crop_type="absolute", crop_size=cfg.INPUT.PATCH_SIZE),
                    dF.RandomFlip(),
                    dF.RandomBrightness(intensity_min=1. - 0.126, intensity_max=1. + 0.126),
                    dF.RandomSaturation(intensity_min=1. - 0.5, intensity_max=1. + 0.5)
                ]
            )  # tT.ColorJitter(brightness=0.126, saturation=0.5)
            if is_train
            else dF.AugmentationList(
                [
                    dF.Resize(cfg.INPUT.FULL_SIZE_TEST),
                    dF.CenterCrop(crop_type="absolute", crop_size=cfg.INPUT.PATCH_SIZE_TEST),
                ]
            )
        )
        self.img_fmt = cfg.INPUT.FORMAT
        self.norm = tT.Normalize(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)

    def __call__(self, img_dict):
        """
        img_dict:
            {
                "img": image path,
                "subclass": subclass index,
                "superclass": superclass index,
                "category":

        """
        result_dict = copy.deepcopy(img_dict)
        img_arr = cv2.imread(img_dict["img"])[:, :, ::-1]

        aug_input = dF.AugInput(img_arr)
        aug_trans = self.augs(aug_input)
        img = aug_input.image
        img = self.norm(tF.to_tensor(img.copy()))
        result_dict["img"] = img

        return result_dict