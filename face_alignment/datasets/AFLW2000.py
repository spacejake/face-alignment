from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io
from scipy import io as sio

import torch
import torch.utils.data as data

from face_alignment.datasets.common import Split, Target, compute_laplacian
from face_alignment.utils import shuffle_lr, flip, crop, getTransform, transform, draw_gaussian, get_preds_fromhm
from face_alignment.util.imutils import *
from face_alignment.util.heatmap import make_gauss, heatmaps_to_coords


from face_alignment.datasets.W300LP import W300LP

class AFLW2000(W300LP):

    def __init__(self, args, split):
        super(AFLW2000, self).__init__(args, split)
        self.is_train = False
        assert self.pointType == '3D', "AFLW2000 provided only 68 3D points"


    def load_extras(self):
        # Don't load extras, will only use this dataset for Validation, for now...
        self.laplcian = None
        pass

    def _getDataFaces(self, is_train):
        base_dir = self.img_dir
        lines = []
        files = [f for f in os.listdir(base_dir) if f.endswith('.mat')]
        for f in files:
            lines.append(os.path.join(base_dir, f))
        print('=> loaded AFLW2000 set, {} images were found'.format(len(lines)))
        return sorted(lines)

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = sio.loadmat(self.anno[idx])
        raw_pts = main_pts['pt3d_68'][0:3, :].transpose()
        raw_pts = torch.from_numpy(raw_pts)
        mins_ = torch.min(raw_pts, 0)[0].view(3) # min vals
        maxs_ = torch.max(raw_pts, 0)[0].view(3) # max vals
        c = torch.FloatTensor((maxs_[0]-(maxs_[0]-mins_[0])/2, maxs_[1]-(maxs_[1]-mins_[1])/2))
        c[1] -= ((maxs_[1]-mins_[1]) * 0.12).float()
        s = (maxs_[0]-mins_[0]+maxs_[1]-mins_[1])/195

        img = load_image(self.anno[idx][:-4] + '.jpg')

        return self.genData(img, raw_pts, c, s, sf, rf)


if __name__=="__main__":
    import face_alignment.util.opts as opts

    args = opts.argparser()
    args.data = "../../data/AFLW2000"
    datasetLoader = AFLW2000
    crop_win = None
    loader = torch.utils.data.DataLoader(
        datasetLoader(args, 'test'),
        batch_size=1,
        #shuffle=True,
        num_workers=1,
        pin_memory=True)
    for i, data in enumerate(loader):
        input, label, meta = data
        target = Target._make(label)
        show_joints3D(target.pts.squeeze(0))
        show_joints(input.squeeze(0), target.pts.squeeze(0))
        show_heatmap(target.heatmap64)
        show_heatmap(target.heatmap256)

        # TEST 256 heatmap extraction
        test_hmpred = heatmaps_to_coords(target.heatmap256)
        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        # TEST 64 heatmap extraction
        test_hmpred = heatmaps_to_coords(target.heatmap64)
        test_hmpred = test_hmpred * 4 # 64->256
        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        plt.pause(0.5)
        plt.draw()
