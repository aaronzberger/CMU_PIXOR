import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_config


class PIXOR_Loss(nn.Module):
    def __init__(self):
        super(PIXOR_Loss, self).__init__()
        config, _, _, _ = load_config()
        self.alpha = config['loss_params']['alpha']
        self.beta = config['loss_params']['beta']
        self.gamma = config['loss_params']['gamma']


    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        # print(preds.shape, targets.shape)
        cls_targets, reg_targets = torch.split(targets, [1, 6], dim=1)
        # print(cls_targets.shape, reg_targets.shape)
        # print('cls targets', torch.max(cls_targets), torch.min(cls_targets))
        # print('reg targets', torch.max(reg_targets), torch.min(reg_targets))

        if preds.size(1) == 7:
            cls_preds, reg_preds = torch.split(preds, [1, 6], dim=1)
        elif preds.size(1) == 15:
            cls_preds, reg_preds, _ = torch.split(preds, [1, 6, 8], dim=1)

        # print('cls preds', torch.max(cls_preds), torch.min(cls_preds))
        # print('reg preds', torch.max(reg_preds), torch.min(reg_preds))
        ################################################################
        # cls_loss = self.focal_loss(cls_preds, cls_targets)
        ################################################################
        cls_loss = cross_entropy(cls_preds, cls_targets) * self.alpha
        ################################################################
        # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
        ################################################################

        pos_pixels = cls_targets.sum()
        if pos_pixels > 0:
            loc_loss = F.smooth_l1_loss(cls_targets * reg_preds, reg_targets, reduction='sum') / pos_pixels * self.beta
            return cls_loss + loc_loss, cls_loss.item(), loc_loss.item()
        else:
            return cls_loss, cls_loss.item(), 0.0


def focal_loss(alpha, gamma, input, target):
    # Inputs must be [0:1] for BCE input
    input = torch.sigmoid(input)

    ce_loss = cross_entropy(input, target, reduction='none')

    # BCE = -log(pt) so pt = e^-BCE
    pt = torch.exp(-ce_loss)

    alpha_tensor = torch.where(target == 1, alpha, 1 - alpha)

    focal_loss = alpha_tensor * (1 - pt)**gamma * ce_loss

    return focal_loss.mean()


def cross_entropy(input, target, weight=None, reduction='mean'):
    return F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
