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


from face_alignment.datasets.W300LP import W300LP

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

class AFLW2000(W300LP):

    def __init__(self, args, split, use_roi=False):
        super(AFLW2000, self).__init__(args, split)
        self.is_train = False
        assert self.pointType == '3D', "AFLW2000 provided only 68 3D points"

        self.std_size = 120
        self.use_roi = use_roi


    def load_extras(self):

        # self.roi_boxes = _load(os.path.join(self.img_dir, 'AFLW2000-3D_crop.roi_box.npy'))
        self.roi_boxes = None


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

        # if self.use_roi and self.roi_boxes is not None:
        #     sx, sy, ex, ey = self.roi_boxes[idx]
        #     s = torch.FloatTensor(((ex - sx) / self.std_size, (ey - sy) / self.std_size))
        #     c = torch.FloatTensor((ex-(ex-sx)/2, ey-(ey-sy)/2))
        #     c[1] -= ((ey-sy) * 0.12).float()
        # else:
        mins_ = torch.min(raw_pts, 0)[0].view(3) # min vals
        maxs_ = torch.max(raw_pts, 0)[0].view(3) # max vals
        c = torch.FloatTensor((maxs_[0]-(maxs_[0]-mins_[0])/2, maxs_[1]-(maxs_[1]-mins_[1])/2))
        c[1] -= ((maxs_[1]-mins_[1]) * 0.12).float()
        # s = torch.FloatTensor((maxs_[0]-mins_[0]+maxs_[1]-mins_[1])/195)
        s = (maxs_[0]-mins_[0]+maxs_[1]-mins_[1])/195

        img = load_image(self.anno[idx][:-4] + '.jpg')

        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = flip(img).float()
                raw_pts = shuffle_lr(raw_pts, width=img.size(2))
                c[0] = img.size(2) - c[0]

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = im_to_torch(crop(im_to_numpy(img), c, s, 256, rotate=r))
        # Transform Points
        # 256x256 GT Heatmap and Points
        pts = raw_pts.clone()
        heatmap256 = torch.zeros(self.nParts, 256, 256)
        transMat256 = getTransform(c, s, 256, rotate=r)
        for i in range(self.nParts):
            if pts[i, 0] > 0:
                pts[i] = transform(pts[i], transMat256)
                pts[i, :2] = pts[i, :2]-1
                heatmap256[i], self.g256 = draw_gaussian(heatmap256[i], pts[i, 0:2], 2, g=self.g256)
                # heatmap256[i] = draw_labelmap(heatmap256[i], pts[i], sigma=3)

        # inp = color_normalize(inp, self.mean, self.std)

        # 64x64 Intermediate Heatmap
        tpts = raw_pts.clone()
        heatmap64 = torch.zeros(self.nParts, 64, 64)
        transMat64 = getTransform(c, s, 64, rotate=r)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                tpts[i] = transform(tpts[i], transMat64)
                heatmap64[i], self.g64 = draw_gaussian(heatmap64[i], tpts[i, 0:2]-1, 1, g=self.g64)
                # heatmap64[i] = draw_labelmap(heatmap64[i], tpts[i] - 1, sigma=1)

        # Compute Target Laplacian vectors
        # lap_pts = compute_laplacian(self.laplcian, pts)

        #return inp, heatmap64, heatmap256, pts, lap_pts, c, s
        return inp.float(), heatmap64.float(), heatmap256.float(), pts.float(), c.float(), s.float()


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
        test_hmpred, _ = get_preds_fromhm(target.heatmap256, target.center, target.scale)
        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        # TEST 64 heatmap extraction
        test_hmpred, _ = get_preds_fromhm(target.heatmap64, target.center, target.scale)
        test_hmpred = test_hmpred * 4 # 64->256
        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        plt.pause(0.5)
        plt.draw()
