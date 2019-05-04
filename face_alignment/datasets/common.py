
from enum import Enum
from typing import NamedTuple

import torch


class Split(Enum):
    train = 'train'
    test = 'test'

class Target(NamedTuple):
    heatmap64: torch.tensor
    heatmap256: torch.tensor
    pts:  torch.tensor
    lap_pts:  torch.tensor
    center: torch.tensor
    scale: torch.tensor

def compute_laplacian(laplacianMat, points):
    lap_pts = torch.matmul(laplacianMat, points)

    return lap_pts