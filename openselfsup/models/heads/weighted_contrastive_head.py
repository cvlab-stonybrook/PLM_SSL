import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class WeightedContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(WeightedContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, pos, neg, weights=None):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        raw_loss = self.criterion(logits, labels)
        if weights is not None:
            # raw_loss = torch.sum(raw_loss * weights) / weights.sum()
            raw_loss = torch.mean(raw_loss * weights)
        else:
            raw_loss = torch.mean(raw_loss)
        losses['loss'] = raw_loss
        return losses
