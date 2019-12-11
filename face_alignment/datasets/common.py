import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.data as data
import numpy as np
import bisect

import skimage
import random
import warnings

from enum import Enum
from typing import NamedTuple



class Split(Enum):
    train = 'train'
    test = 'test'


class Target(NamedTuple):
    heatmap64: torch.tensor
    heatmap_eyes: torch.tensor
    pts: torch.tensor
    pts64: torch.tensor
    lap_pts: torch.tensor
    center: torch.tensor
    scale: torch.tensor
    has_3d_anno: torch.tensor
    has_2d_anno: torch.tensor
    pts_2d: torch.tensor

def boolToTensor(bool_val):
    return torch.ones(1).byte() if bool_val else torch.zeros(1).byte()

def compute_laplacian(laplacianMat, points):
    lap_pts = torch.matmul(laplacianMat, points)

    return lap_pts


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW', unnorm=False):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel
        self.unnorm = unnorm

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = Parameter(torch.ones(1))

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        feature = feature*30 # 30 is purely empirical

        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)

        if self.unnorm:
            w = float(self.width) - 1
            h = float(self.height) - 1
            expected_x = (expected_x * w + w) / 2.
            expected_y = (expected_y * h + h) / 2.

        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


def compute_laplacian(laplacianMat, points):
    lap_pts = torch.matmul(laplacianMat, points)

    return lap_pts

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = 1.0 - np.sqrt(1.0-lam)
    cut_w = np.int(W * cut_rat[0])
    cut_h = np.int(H * cut_rat[1])

    # TODO: add option for uniform and normal
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1, bby1, bbx2, bby2 = gen_bbox((cut_w, cut_h), (cx, cy))

    bbx1 = np.clip(bbx1, 0, W)
    bby1 = np.clip(bby1, 0, H)
    bbx2 = np.clip(bbx2, 0, W)
    bby2 = np.clip(bby2, 0, H)

    return bbx1, bby1, bbx2, bby2

def gen_bbox(size, center):
    cx, cy = center
    w = size[0]
    h = size[1]

    bbx1 = int(cx - w // 2)
    bby1 = int(cy - h // 2)
    bbx2 = int(cx + w // 2)
    bby2 = int(cy + h // 2)

    return bbx1, bby1, bbx2, bby2

def cutout(img, low=0.4, high=0.5):
    # Perform cutout
    lam = np.random.uniform(low, high, 2)
    # TODO: use normal center like cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.unsqueeze(0).size(), lam)
    img[:, bbx1:bbx2, bby1:bby2] = 0

def eye_occlusion(img, center, low=0.4, high=0.6):
    # Perform cutout
    size = img.unsqueeze(0).size()
    w_rat = np.random.normal(0.2, 0.01, 1)
    h_rat = np.random.normal(0.6, 0.05, 1)
    W = size[2]
    H = size[3]
    cut_w = np.int(W * w_rat)
    cut_h = np.int(H * h_rat)

    cx = center[1]
    cy = center[0]

    bbx1, bby1, bbx2, bby2 = gen_bbox((cut_w, cut_h), (cx, cy))

    img[:, bbx1:bbx2, bby1:bby2] = 0

def cutmix(img, cut_img, ratio=(0.2, 0.6), m=(310, 220), sigma=(40,50)):
    # generate mixed sample

    # make random box cut from cut_img
    lam = np.random.uniform(ratio[0], ratio[1], 2)
    cbbx1, cbby1, cbbx2, cbby2 = rand_bbox(img.unsqueeze(0).size(), lam)
    cut_img = cut_img[..., cbbx1:cbbx2, cbby1:cbby2]

    # Place in image, with normal distribution from face center
    w, h = (cbbx2 - cbbx1), (cbby2 - cbby1)
    rx, ry = w // 2, h // 2
    ccx = (cbbx1 + cbbx2) // 2
    ccy = (cbby1 + cbby2) // 2
    cx_bounds = (rx, img.size(1) - rx - 1)
    cy_bounds = (ry, img.size(2) - ry - 1)

    cx = np.clip(int(np.random.normal(310, 40)), cx_bounds[0], cx_bounds[1])
    cy = np.clip(int(np.random.normal(220, 50)), cy_bounds[0], cy_bounds[1])

    dx = cx - ccx
    dy = cy - ccy

    bbx1 = dx + cbbx1
    bby1 = dy + cbby1
    bbx2 = dx + cbbx2
    bby2 = dy + cbby2

    img[..., bbx1:bbx2, bby1:bby2] = cut_img

    return img

class ConcatDataset(data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # for d in self.datasets:
        #     assert not isinstance(d, data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

# class ChainDataset(data.IterableDataset):
#     r"""Dataset for chainning multiple :class:`IterableDataset` s.
#
#     This class is useful to assemble different existing dataset streams. The
#     chainning operation is done on-the-fly, so concatenating large-scale
#     datasets with this class will be efficient.
#
#     Arguments:
#         datasets (iterable of IterableDataset): datasets to be chained together
#     """
#     def __init__(self, datasets):
#         super(ChainDataset, self).__init__()
#         self.datasets = datasets
#
#     def __iter__(self):
#         for d in self.datasets:
#             assert isinstance(d, data.IterableDataset), "ChainDataset only supports IterableDataset"
#             for x in d:
#                 yield x
#
#     def __len__(self):
#         total = 0
#         for d in self.datasets:
#             assert isinstance(d, data.IterableDataset), "ChainDataset only supports IterableDataset"
#             total += len(d)
#         return total

if __name__ == '__main__':
    from face_alignment.utils import draw_gaussian, draw_gaussianv2
    from skimage.filters import gaussian
    import matplotlib.pyplot as plt
    from random import randint

    n, c, h, w = 2, 1, 256, 256
    vis = True

    # Generate fake heatmap with random keypoint locations
    hm = torch.zeros((n, c, h, w))

    random_keypoints = []
    for i in range(n):
        batch = []
        for j in range(c):
            kps = [np.random.randint(w), np.random.randint(h)]
            # hm[i, j, kps[1], kps[0]] = 1.
            batch.append(kps[0])
            batch.append(kps[1])
        random_keypoints.append(batch)

    # Put gaussian to peaks
    random_keypoints = np.array(random_keypoints)
    random_keypoints = torch.from_numpy(random_keypoints)
    for i in range(n):
        for j in range(c):
            # hm[i, j] = torch.from_numpy(gaussian(hm[i, j].numpy(), sigma=1.))
            # hm[i, j] = (hm[i, j] / (hm[i, j].max() + 1e-7)) * 30.  # 30 is purely empirical
            hm[i, j] = draw_gaussianv2(hm[i, j], random_keypoints[i, 2*j:2*j+2].long(), sigma=2.)
            if vis:
                plt.imshow(hm[i, j])
                plt.show()

    layer = SpatialSoftmax(h, w, c, temperature=1., unnorm=True)

    if vis:
        pos = torch.cat([layer.pos_x.reshape(h, w), layer.pos_y.reshape(h, w)], 1)
        plt.imshow(pos.numpy())
        plt.show()

    keypoints = layer(hm).round().int()

    random_keypoints = random_keypoints.int()

    print('Original kps: %s' % random_keypoints)
    print('Estimated kps: %s' % keypoints)
    print('Difference: %s' % ((random_keypoints - keypoints) ** 2).sum().item())
