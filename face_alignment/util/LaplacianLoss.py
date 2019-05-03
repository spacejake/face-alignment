"""
Source Code originally from https://github.com/ThibaultGROUEIX/3D-CODED
"""

import torch
import numpy as np

from face_alignment.util.laplacian import Laplacian

class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, toref=True):
        # Input:
        #  faces: F x 3
        self.toref = toref
        # V x V
        self.laplacian = Laplacian(faces)
    
    def __call__(self, pred, target):
        Lx_gt = self.laplacian(target)
        self.curve_gt = torch.norm(Lx_gt.view(-1, Lx_gt.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt*0

        Lx = self.laplacian(pred)
        # Reshape to BV x 3
        Lx = Lx.view(-1, Lx_gt.size(2))
        loss = torch.abs(torch.norm(Lx, p=2, dim=1).float()-self.curve_gt).mean()

        return loss
