import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)

class OrthRegRegularization(nn.Module):
    def __init__(self, base_channels=2048, sigma=2.0):
        super(OrthRegRegularization, self).__init__()
        self.sigma = sigma
        self.base_channels = base_channels
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, rofs, centers):
        """
        rofs: region orthogonal feature,
              B x M x C
        centers: B x M x C
        """
        center_loss = self.calculate_center(rofs, centers)
        orth_loss = self.calculate_orth(rofs, centers)
        return center_loss, orth_loss

    def calculate_center(self, rofs, centers):
        return self.l2_loss(rofs, centers) / rofs.size(0)

    def calculate_orth(self, rofs, centers):
        N, MC = rofs.size()
        rofs = rofs.reshape(N, MC//self.base_channels, self.base_channels)  # N, M, C
        centers = centers.reshape(N, MC//self.base_channels, self.base_channels)  # N, M, C
        rofs = F.normalize(rofs, p=2, dim=-1)
        centers = F.normalize(centers, p=2, dim=-1)
        centers = centers.permute(0, 2, 1).contiguous()  # N,M,C -> N,C,M

        batch_similarity_matrix = torch.bmm(rofs, centers) # N, M, M

        mask = torch.eye(batch_similarity_matrix.size(1), batch_similarity_matrix.size(2)).bool().to(batch_similarity_matrix.device) # M, M
        mask = (~mask).float().unsqueeze(0) # 1, M, M

        batch_orth_matrix = torch.abs(batch_similarity_matrix*mask)
        orth_loss = batch_orth_matrix.sum() / (mask.sum() + 1e-6) / N

        return orth_loss


class CenOrthRegRegularization(nn.Module):
    def __init__(self, base_channels=2048, sigma=2.0):
        super(CenOrthRegRegularization, self).__init__()
        self.sigma = sigma
        self.base_channels = base_channels
        # self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, rofs, centers):
        """
        rofs: region orthogonal feature,
              B x M x C
        centers: B x M x C
        """

        N, MC = rofs.size()
        rofs = rofs.reshape(N, MC//self.base_channels, self.base_channels)  # N, M, C
        centers = centers.reshape(N, MC//self.base_channels, self.base_channels)  # N, M, C
        rofs = F.normalize(rofs, p=2, dim=-1)
        centers = F.normalize(centers, p=2, dim=-1)
        centers = centers.permute(0, 2, 1).contiguous()  # N,M,C -> N,C,M

        batch_similarity_matrix = torch.bmm(rofs, centers) # N, M, M

        mask = torch.eye(batch_similarity_matrix.size(1), batch_similarity_matrix.size(2)).bool().to(batch_similarity_matrix.device) # M, M
        mask_pos = mask.float().unsqueeze(0)  # 1, M, M
        mask_neg = (~mask).float().unsqueeze(0)  # 1, M, M

        batch_cen_matrix = batch_similarity_matrix * mask_pos
        center_loss = 1.0 - batch_cen_matrix.sum() / (mask_pos.sum() + 1e-6) / N

        batch_orth_matrix = torch.abs(batch_similarity_matrix*mask_neg)
        orth_loss = batch_orth_matrix.sum() / (mask_neg.sum() + 1e-6) / N

        return center_loss, orth_loss