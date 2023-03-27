import torch
from torch import nn

from .registry import NECKS
from .utils import build_norm_layer
from .necks import _init_weights
from .utils import cal_intersection_batch


@NECKS.register_module
class NonLinearNeckV1Dense(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 num_grid=None):
        super(NonLinearNeckV1Dense, self).__init__()
        self.with_avg_pool = with_avg_pool
        # if with_avg_pool:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x, dense=False):
        #assert len(x) == 1
        if dense:
            # avgpooled_x = x[-1]
            avgpooled_x = self.avgpool(x)
            avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

            if self.with_pool:
                x = self.pool(x) # sxs
            x = self.mlp2(x)  # sxs: bxdxsxs
            avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
            x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
            avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
            return [avgpooled_x, x, avgpooled_x2]
        else:
            x = x[-1]
            if self.with_avg_pool:
                x = self.avgpool(x)
            return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class MatchingNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 linear=True):
        super(MatchingNeck, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.linear = linear

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forard_linear(self, x):
        return self.fc(x)

    def forward(self, k_grid, matching_mat_q):
        # print("k_grid:", k_grid.shape, "matching_mat_q:", matching_mat_q.shape)
        k_grid = torch.bmm(matching_mat_q, k_grid)
        if self.linear:
            k_grid = self.fc(k_grid)
        return k_grid


@NECKS.register_module
class AttentionMatchingNeck(nn.Module):
    '''
    '''
    def __init__(self,
                 in_channels,
                 mid_dim,
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 linear=True):
        super(AttentionMatchingNeck, self).__init__()
        # self.fc = nn.Linear(in_channels, out_channels)
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, mid_dim, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.Conv1d(mid_dim, 1, kernel_size=kernel_size, padding=padding)
        )
        self.fc = nn.Linear(in_channels, out_channels)
        self.linear = linear

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forard_linear(self, x):
        return self.fc(x)

    def forward(self, k_grid, matching_mat_q):

        mask = matching_mat_q > 0

        x = k_grid.permute(0, 2, 1)  # BB, dim, att_n

        A = self.attention(x)  # BB, 1, att_n
        b, _, att_n = A.shape
        A = A.repeat(1, att_n, 1) # BB, att_n, att_n

        A = masked_softmax(A, mask)

        # print("k_grid:", k_grid.shape, "matching_mat_q:", matching_mat_q.shape)
        k_grid = torch.bmm(A, k_grid)
        if self.linear:
            k_grid = self.fc(k_grid)
        return k_grid


def non_zero_divide(a, b):
    """
    a.shape = b, ...
    b.shape = b,
    Args:
        a:
        b:

    Returns:
        c = a / b, c = 0 where b = 0
    """
    c = torch.zeros_like(a)
    mask = (b > 0.)
    c[mask, ...] = a[mask, ...] / b[mask, None]
    return c


def masked_softmax(attention, mask):
    """
    Args:
        attention: shape: bb, att_n, att_n
        mask: the same

    Returns:

    """
    # from torch.nn import functional as F
    # F.softmax()
    attention = attention.exp() * mask
    b, att_n, att_m = attention.shape
    att_sum = torch.sum(attention, dim=-1)
    attention = non_zero_divide(attention.flatten(0, 1), att_sum.flatten(0, 1))
    attention = attention.reshape(b, att_n, att_m)
    return attention
