import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
from chrf.layers import ShapeSpec
from chrf.modeling.backbone import build_backbone
from chrf.modeling.necks import build_neck
from chrf.modeling.heads import build_head
from chrf.modeling.loss import *
from chrf.utils.events import get_event_storage
from chrf.utils.visualizer import pca_feat
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as tF

from ..build import META_ARCH_REGISTRY

EPS = 1e-15

@META_ARCH_REGISTRY.register()
class CHRFClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_features = cfg.BENCHMARK.IN_FEATURES
        self.backbone = build_backbone(cfg)
        self.hiers = cfg.BENCHMARK.HIERARCHY

        output_shapes = self.backbone.output_shape()
        bk_out_shape = output_shapes[self.in_features[-1]]
        neck_in_shape = ShapeSpec(
            channels=bk_out_shape.channels,
            height=bk_out_shape.height,
            width=bk_out_shape.width,
            stride=bk_out_shape.stride
        )
        self.neck = build_neck(cfg, neck_in_shape)
        for item in self.neck:
            self.add_module(item+'_neck', self.neck[item])

        head_in_ch, head_in_h, head_in_w, head_in_s = (
            {hier: self.neck[hier].out_channels for hier in self.hiers},
            bk_out_shape.height,
            bk_out_shape.width,
            bk_out_shape.stride,
        )
        head_in_shape = ShapeSpec(
            channels=head_in_ch, height=head_in_h, width=head_in_w, stride=head_in_s
        )

        self.head = build_head(cfg, head_in_shape)
        for hier in self.head:
            self.add_module(hier+'_classifyHead', self.head[hier])

        # add attention regularization with center loss for each hierarchy
        for category_num, hier in zip(cfg.MHBENCHMARK.CLASSIFICATION.CATEGORY_NUM, self.hiers):
            self.register_buffer(
                hier+'_feature_center',
                torch.zeros(
                    category_num,
                    head_in_ch[hier],
                    requires_grad=False,
                )
            )

        self.center_beta = cfg.BENCHMARK.CENTER_BETA
        self.ORR_lambda = cfg.BENCHMARK.ORR_LAMBDA
        self.ORR = cfg.BENCHMARK.OrthRegRegularization
        self.sigma = cfg.BENCHMARK.SIGMA
        assert isinstance(self.sigma, list) and len(self.hiers) == len(self.sigma)

        if self.ORR == "OR":
            self.center_loss = OrthRegRegularization(base_channels=neck_in_shape.channels)
        elif self.ORR == "COR":
            self.center_loss = CenOrthRegRegularization(base_channels=neck_in_shape.channels)
        elif self.ORR == "C":
            self.center_loss = CenterLoss()
        else:
            raise Exception("{} do not exist ORR type!".format(self.ORR))

        self.use_gmap = cfg.BENCHMARK.LOAD_GAZE
        self.same_gmaps = cfg.BENCHMARK.KEEP_SAME_GAZE_NUM
        if self.use_gmap:
            self.gaze_loss = None

        self.strides = [
            output_shapes[in_feature].stride for in_feature in self.in_features
        ]

        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        )
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.vis_period = cfg.VIS_PERIOD

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        # preprocess, obtain img, target and gaze map
        img_t_list = [item["img"] for item in batched_inputs]
        batched_img_t = default_collate(img_t_list).to(self.device)
        batched_tag_t = {
            hier: torch.tensor(
                [in_item[hier] for in_item in batched_inputs], device=self.device
            ) for hier in self.hiers
        }

        if self.use_gmap:
            gmap_t_list = [item["gmaps"] for item in batched_inputs]
            batched_gmaps_t = {
                hier: [] for hier in self.hiers
            }
            for gaze_img in gmap_t_list:
                for hier, gmap in gaze_img.items():
                    batched_gmaps_t[hier].append(gmap)
            if self.same_gmaps:
                for hier in self.hiers:
                    heatmap_list = batched_gmaps_t[hier]
                    heatmap_num_per_img = [len(map) for map in heatmap_list]
                    max_heatmap_num = max(heatmap_num_per_img)
                    batched_gmaps_t[hier] = [self.mapRepeat(heatmap, max_heatmap_num) for heatmap in heatmap_list]
        else:
            batched_gmaps_t = None

        orign_losses, multih_att, orign_res, center_loss = self.process(batched_img_t, batched_tag_t, batched_gmaps_t=batched_gmaps_t, with_center=True)
        attention_map = multih_att[self.hiers[0]]  # just use the most fine-grained level attention

        if not self.training:
            test_aug_img = self.batch_augment(batched_img_t, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            aug_losses, aug_attention_map, aug_res, _ = self.process(test_aug_img, batched_tag_t, batched_gmaps_t=None, with_center=False)
            res = []
            for orign_r, aug_r in zip(orign_res, aug_res):
                res.append((orign_r+aug_r)/2.)
            return res

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = self.batch_augment(batched_img_t, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = self.batch_augment(batched_img_t, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        aug_targets = {
            hier: torch.cat([batched_tag_t[hier], batched_tag_t[hier]], dim=0) for hier in self.hiers
        }
        aug_losses, aug_attention_map, aug_res, _ = self.process(aug_images, aug_targets, batched_gmaps_t=None, with_center=False)

        losses = {
            loss_name: orign_losses[loss_name]/3. + aug_losses[loss_name]*2./3. for loss_name in orign_losses
        }
        losses.update(center_loss)

        return losses

    def process(self, batched_img_t, batched_tag_t, batched_gmaps_t=None, with_center=True):
        multih_fg_map, multih_fmatrixs, multih_scores, multih_att, multih_atts = self.compute_score(batched_img_t)
        if with_center:
            center_loss = self.update_feature_center(multih_fmatrixs, batched_tag_t)
        else:
            center_loss = None
        del multih_fg_map, multih_fmatrixs
        ## cross entropy loss
        multih_loss, multih_acc = self.compute_loss(batched_tag_t, multih_scores)

        ## gaze and attention loss
        if self.use_gmap and batched_gmaps_t is not None:
            gaze_attention_loss = self.compute_gaze_attention_loss(batched_gmaps_t, multih_atts)

        if not self.training:
            res = []
            for hier in self.hiers:
                res.append(multih_scores[hier])
                res.append(multih_acc[hier])
            return None, multih_att, res, None

        losses = {}
        for hier in multih_loss:
            for loss_name in multih_loss[hier]:
                losses[hier + '_' + loss_name] = multih_loss[hier][loss_name]
        return losses, multih_att, None, center_loss


    def compute_score(self, batched_inputs):
        batch_size = batched_inputs.size(0)

        multih_fmap = self.backbone(batched_inputs)
        multih_fmap = {hier: feats[self.in_features[-1]] for hier, feats in multih_fmap.items()}

        multih_fmatrixs, multih_scores, multih_att, multih_atts = {}, {}, {}, {}

        shallow_feature_matrix = None
        for hier in reversed(self.hiers):
            feature_matrix, attention_maps = self.neck[hier](multih_fmap[hier])
            # aggregate single hierarchy feature
            # multih_fmatrixs[hier] = feature_matrix
            shallow_feature_matrix, feature_matrix = self.neck[hier].dohf(shallow_feature_matrix, feature_matrix)
            scores = self.head[hier](feature_matrix * 100.)
            # aggregate dohf hierarchy feature
            multih_fmatrixs[hier] = feature_matrix
            multih_scores[hier] = scores

            # Generate Attention Map
            if self.training:
                # Randomly choose one of attention maps Ak
                attention_map = []
                for i in range(batch_size):
                    attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + self.neck[hier].EPSILON)
                    attention_weights = F.normalize(attention_weights, p=1, dim=0)
                    k_index = np.random.choice(self.neck[hier].M, 2, p=attention_weights.cpu().numpy())
                    attention_map.append(attention_maps[i, k_index, ...])
                attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
            else:
                attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

            multih_att[hier] = attention_map
            multih_atts[hier] = attention_maps

        return multih_fmap, multih_fmatrixs, multih_scores, multih_att, multih_atts

    def compute_loss(self, batched_tag_t, multih_scores):
        multih_loss, multih_acc = {}, {}
        for hier in self.hiers:
            targets = batched_tag_t[hier]
            scores = multih_scores[hier]
            # log_prob = torch.log(scores + EPS), Softmax has been removed from ClassificationHead
            loss = F.cross_entropy(scores, targets)
            with torch.no_grad():
                top1_acc = (
                    torch.argmax(scores, dim=1).view(-1) == targets
                ).sum().float().item() / targets.shape[0]
            multih_loss[hier] = {"cross_entropy": loss}
            multih_acc[hier] = top1_acc

        return multih_loss, multih_acc

    def update_feature_center(self, feature_matrix, batched_tag_t):
        multih_center_loss = {}
        for hier, sigma in zip(self.hiers, self.sigma):
            feature_center = getattr(self, hier+'_feature_center')
            y = batched_tag_t[hier]
            feature_center_batch = F.normalize(feature_center[y], dim=-1)
            feature_center[y] += self.center_beta * (feature_matrix[hier].detach() - feature_center_batch)
            if self.ORR == "OR" or self.ORR == "COR":
                center_loss, orth_loss = self.center_loss(feature_matrix[hier], feature_center_batch)
                multih_center_loss[hier + '_center_loss'] = center_loss * self.ORR_lambda
                multih_center_loss[hier + '_orth_loss'] = orth_loss * sigma * self.ORR_lambda
            elif self.ORR == "C":
                center_loss = self.center_loss(feature_matrix[hier], feature_center_batch)
                multih_center_loss[hier+'_center_loss'] = center_loss * self.ORR_lambda
        return multih_center_loss

    def compute_gaze_attention_loss(self, batched_gmaps_t, multih_atts):
        return None

    def mapRepeat(self, map, num):
        """
        map: tensor C x H x W
        """
        map_num = len(map)
        if map_num == num:
            return map
        elif map_num > num:
            return map[:num].clone()
        else:
            ratio = num // map_num + 1
            map = map.repeat(ratio, 1, 1)
            return map[:num].clone()

    def build_pooling(self, cfg):
        mode = cfg.BENCHMARK.CLASSIFICATION.POOLING_MODE
        if mode == "avg":
            return nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif mode == "max":
            return nn.AdaptiveMaxPool2d(output_size=(1, 1))
        else:
            raise ValueError("Unknown pooling mode: {}".format(mode))

    def batch_augment(self, images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
        batches, _, imgH, imgW = images.size()

        if mode == 'crop':
            crop_images = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = theta * atten_map.max()

                crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
                nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

                crop_images.append(
                    F.upsample_bilinear(
                        images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=(imgH, imgW)))
            crop_images = torch.cat(crop_images, dim=0)
            return crop_images

        elif mode == 'drop':
            drop_masks = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_d = theta * atten_map.max()

                drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
            drop_masks = torch.cat(drop_masks, dim=0)
            drop_images = images * drop_masks.float()
            return drop_images

        else:
            raise ValueError(
                'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)