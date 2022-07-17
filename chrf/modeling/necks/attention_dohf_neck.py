import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import NECKS_REGISTRY

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class AttentionDohfNeck2(nn.Module):

    def __init__(self, M=32, res_channels=2048, pooling_mode='GAP', add_lambda=0.8):
        super(AttentionDohfNeck2, self).__init__()
        self.M = M
        self.base_channels = res_channels
        self.out_channels = M * res_channels
        self.conv = BasicConv2d(res_channels, self.M, kernel_size=1)

        self.pooling = self.build_pooling(pooling_mode)
        self.EPSILON = 1e-6

        self.add_lambda = add_lambda

    def build_pooling(self, pooling_mode):
        if pooling_mode == "GAP":
            return None
        elif pooling_mode == "GMP":
            return nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("Unknown pooling mode: {}".format(pooling_mode))

    def bilinear_attention_pooling(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pooling is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pooling(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        return feature_matrix

    def forward(self, x):

        attention_maps = self.conv(x)
        feature_matrix = self.bilinear_attention_pooling(x, attention_maps)

        return feature_matrix, attention_maps

    def dohf(self, shallow_hiera, deep_hiera):
        """
        from shallow to deep: order, family, genus, class
        shallow_hiera: N, M*C
        deep_hiera: N, M*C

        return
        """
        if shallow_hiera==None:
            return deep_hiera, deep_hiera
        N1, MC1 = shallow_hiera.shape
        M1 = MC1//self.base_channels
        shallow_hiera_mean = shallow_hiera.reshape(N1, M1, self.base_channels) # N,M1*C -> N,M1,C
        shallow_hiera_mean = shallow_hiera_mean.mean(dim=1) # N, C

        N2, MC2 = deep_hiera.shape
        M2 = MC2//self.base_channels
        deep_hiera_dohf = deep_hiera.reshape(N2, M2, self.base_channels)  # N,M2*C -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous() # N,M2,C -> N,C,M2

        shallow_hiera_norm = torch.norm(shallow_hiera_mean, p=2, dim=1) # N
        projection = torch.bmm(shallow_hiera_mean.unsqueeze(1), deep_hiera_dohf)  # N, 1, M2
        projection = torch.bmm(shallow_hiera_mean.unsqueeze(2), projection)  # N, C, M2
        projection = projection / (shallow_hiera_norm * shallow_hiera_norm).view(-1, 1, 1)  # N, C, M2

        orthogonal_comp = deep_hiera_dohf - projection
        deep_hiera_dohf = deep_hiera_dohf + self.add_lambda * orthogonal_comp # N, C, M2
        deep_hiera_dohf = deep_hiera_dohf.permute(0, 2, 1).contiguous() # N, C, M2 -> N,M2,C
        deep_hiera_dohf = deep_hiera_dohf.reshape(N2, -1) # N,M2,C -> N, MC2
        # l2 normalization along dimension M2 and C
        deep_hiera_dohf = F.normalize(deep_hiera_dohf, dim=-1)
        return deep_hiera, deep_hiera_dohf

@NECKS_REGISTRY.register()
def build_attention_dohf_diffM_multih_neck(cfg, input_shape):
    res_channels = input_shape.channels
    M = cfg.MODEL.NECK.M
    pooling_mode = cfg.MODEL.NECK.POOLING_MODE
    add_lambda = cfg.MODEL.NECK.ADD_LAMBDA
    hiers = cfg.BENCHMARK.HIERARCHY
    assert isinstance(M, list) and len(M)==len(hiers)
    necks = {}

    for hiera, m in zip(hiers, M):
        necks[hiera] = AttentionDohfNeck2(M=m, res_channels=res_channels, pooling_mode=pooling_mode, add_lambda=add_lambda)

    return necks