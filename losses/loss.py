import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    """Dice loss on the foreground (leaf) class.

    Args:
        logits: (B, C, H, W) logits, softmax taken over class dim.
        target: (B, H, W) long tensor with values in {0, 1}.
    Return:
        scalar tensor: 1 - Dice(foreground intersection, union)
    """
    n, c, h, w = logits.size()
    if c == 1:
        prob = torch.sigmoid(logits)
    else:
        prob = F.softmax(logits, dim=1)[:, 1]  # foreground channel
    target = target.float().view(n, -1)
    prob = prob.view(n, -1)

    intersection = (prob * target).sum(dim=1)
    cardinality = prob.sum(dim=1) + target.sum(dim=1)

    return (1.0 - ((2.0 * intersection + smooth) / (cardinality + smooth))).mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce', 'focal', 'ce_dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'ce_dice':
            return self.CrossEntropyDiceLoss
        else:
            raise NotImplementedError(f"Unknown loss mode: {mode}")

    def _ce(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss /= n
        return loss

    def CrossEntropyLoss(self, logit, target):
        return self._ce(logit, target)

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyDiceLoss(self, logit, target):
        """Weighted Cross Entropy + Dice Loss.

        CE provides pixel-level signal; Dice optimizes leaf region overlap.
        Target shape: (B, H, W) with values 0/1.
        Logit shape: (B, 2, H, W).
        """
        ce = self._ce(logit, target)
        dice = dice_loss(logit, target)
        return ce + dice