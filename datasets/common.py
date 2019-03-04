
from enum import Enum
from typing import NamedTuple

import torch


class Split(Enum):
    train = 'train'
    test = 'test'

class Target(NamedTuple):
    heatmap: torch.tensor
    pts:  torch.tensor
    center: torch.tensor
    scale: torch.tensor
