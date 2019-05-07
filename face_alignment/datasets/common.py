import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from enum import Enum
from typing import NamedTuple

from face_alignment.utils import draw_gaussian


class Split(Enum):
    train = 'train'
    test = 'test'


class Target(NamedTuple):
    heatmap64: torch.tensor
    heatmap256: torch.tensor
    pts: torch.tensor
    lap_pts: torch.tensor
    center: torch.tensor
    scale: torch.tensor


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
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.width),
            np.linspace(-1., 1., self.height)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
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
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


if __name__ == '__main__':
    from skimage.filters import gaussian
    import matplotlib.pyplot as plt
    from random import randint

    n, c, h, w = 2, 16, 64, 48
    vis = False

    # Generate fake heatmap with random keypoint locations
    hm = torch.zeros((n, c, h, w))

    random_keypoints = []
    for i in range(n):
        batch = []
        for j in range(c):
            kps = [np.random.randint(w), np.random.randint(h)]
            hm[i, j, kps[1], kps[0]] = 1.
            batch.append(kps[0])
            batch.append(kps[1])
        random_keypoints.append(batch)

    # Put gaussian to peaks
    random_keypoints = np.array(random_keypoints)
    random_keypoints = torch.from_numpy(random_keypoints)
    for i in range(n):
        for j in range(c):
            hm[i, j] = torch.from_numpy(gaussian(hm[i, j].numpy(), sigma=1.))
            # hm[i, j], _ = draw_gaussian(hm[i, j], random_keypoints[i, 2*j:2*j+2].float(), 1)
            hm[i, j] = (hm[i, j] / (hm[i, j].max() + 1e-7)) * 30.  # 30 is purely empirical
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
