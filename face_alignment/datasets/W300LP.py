from __future__ import print_function, division, absolute_import

import os
import numpy as np
import random
import math
from skimage import io
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.utils.data as data

from face_alignment.datasets.common import Split, Target, compute_laplacian
from face_alignment.utils import shuffle_lr, flip, crop, getTransform, transform, draw_gaussian, get_preds_fromhm
from face_alignment.util.imutils import *
from face_alignment.util.evaluation import get_preds, accuracy_points, calc_metrics

'''
Modified derivative of https://github.com/hzh8311/pyhowfar
'''

class W300LP(data.Dataset):

    def __init__(self, args, split='train'):
        self.nParts = 68
        self.pointType = args.pointType
        self.img_dir = args.data
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor

        if self.pointType == '2D':
            self.lmk_dir = os.path.join(self.img_dir, 'landmarks')
        else:
            self.lmk_dir = os.path.join(self.img_dir, 'landmarks3d')

        # self.anno = anno
        self.split = Split(split)
        self.is_train = self.split is Split.train
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.load_extras()

        self.g64 = None
        self.g256 = None

    def load_extras(self):
        self.mean, self.std = self._comput_mean()

    def _getDataFaces(self, is_train):
        base_dir = self.lmk_dir

        dirs = os.listdir(base_dir)
        lines = []
        vallines = []
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.mat')]
            for f in files:
                if 'test' not in f:
                    lines.append(f)
                else:
                    vallines.append(f)
        if is_train:
            print('=> loaded train set, {} images were found'.format(len(lines)))
            return lines
        else:
            print('=> loaded validation set, {} images were found'.format(len(vallines)))
            return vallines

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, heatmap64, heatmap256, pts, center, scale = self.generateSampleFace(index)
        target = Target(heatmap64, heatmap256, pts, center, scale)
        if self.is_train:
            return inp, target
        else:
            meta = {'index': index, 'center': center, 'scale': scale} #, 'pts': pts,}
            return inp, target, meta

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = sio.loadmat(
            os.path.join(self.lmk_dir, self.anno[idx].split('_')[0],
                         self.anno[idx][:-4] + '.mat'))
        raw_pts = main_pts['pts_2d'] if self.pointType == '2D' else main_pts['pts_3d']
        raw_pts = torch.from_numpy(raw_pts)
        c = torch.Tensor((450 / 2, 450 / 2 + 50))
        s = 1.8

        img = load_image(
            os.path.join(self.img_dir, self.anno[idx].split('_')[0], self.anno[idx][:-8] +
                         '.jpg'))

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

        # show_joints(img.squeeze(0), raw_pts.squeeze(0))
        # show_joints(inp.squeeze(0), pts.squeeze(0))
        return inp, heatmap64, heatmap256, pts, c, s

    def _comput_mean(self):
        meanstd_file = os.path.join(self.img_dir, 'mean.pth.tar')
        if os.path.isfile(meanstd_file):
            ms = torch.load(meanstd_file)
        else:
            print("\tcomputing mean and std for the first time, it may takes a while, drink a cup of coffe...")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            if self.is_train:
                for i in range(self.total):
                    a = self.anno[i]
                    img_path = os.path.join(self.img_dir, self.anno[i].split('_')[0],
                                            self.anno[i][:-8] + '.jpg')
                    img = load_image(img_path)
                    mean += img.view(img.size(0), -1).mean(1)
                    std += img.view(img.size(0), -1).std(1)

                mean /= self.total
                std /= self.total
                ms = {
                    'mean': mean,
                    'std': std,
                }
                torch.save(ms, meanstd_file)
        if self.is_train:
            print('\tMean: %.4f, %.4f, %.4f' % (ms['mean'][0], ms['mean'][1], ms['mean'][2]))
            print('\tStd:  %.4f, %.4f, %.4f' % (ms['std'][0], ms['std'][1], ms['std'][2]))
        return ms['mean'], ms['std']

if __name__=="__main__":
    import face_alignment.util.opts as opts

    args = opts.argparser()
    args.data = "../../data/300W_LP"
    # dataset = W300LP(args, Split.test)
    datasetLoader = W300LP
    crop_win = None
    loader = torch.utils.data.DataLoader(
        datasetLoader(args, 'test'),
        batch_size=args.val_batch,
        #shuffle=True,
        num_workers=4,
        pin_memory=True)

    idx = range(1, 69, 1)
    all_dists256 = torch.zeros((68, loader.dataset.__len__()))
    all_dists64 = torch.zeros((68, loader.dataset.__len__()))
    for val_idx, data in enumerate(loader):
        input, label, meta = data
        target = Target._make(label)
        # show_joints3D(target.pts.squeeze(0))
        # show_joints(input.squeeze(0), target.pts.squeeze(0))
        # show_heatmap(target.heatmap64)
        # show_heatmap(target.heatmap256)

        # TEST 256 heatmap extraction
        test_hmpred, _ = get_preds_fromhm(target.heatmap256, target.center, target.scale)
        # show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        # TEST 64 heatmap extraction
        test_hmpred, _ = get_preds_fromhm(target.heatmap64, target.center, target.scale)
        test_hmpred = test_hmpred * 4 # 64->256
        # show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        # plt.pause(0.5)
        # plt.draw()

        # acc256, batch_dists256 = accuracy_points(test_hmpred, target.pts[:,:,:2], idx, thr=0.07)
        # all_dists256[:, val_idx * args.val_batch:(val_idx + 1) * args.val_batch] = batch_dists256

        acc64, batch_dists64 = accuracy_points(test_hmpred, target.pts[:,:,:2], idx, thr=0.07)
        all_dists64[:, val_idx * args.val_batch:(val_idx + 1) * args.val_batch] = batch_dists64


    # mean_error256 = torch.mean(all_dists256)
    mean_error64 = torch.mean(all_dists64)

    # auc256 = calc_metrics(all_dists256, path=args.checkpoint, category='300W-Testset-256',
    #                       method='argmax 256')  # this is auc of predicted maps and target.
    auc64 = calc_metrics(all_dists64, path=args.checkpoint, category='300W-Testset-64',
                         method='argmax 64->256')  # this is auc of predicted maps and target.

    # print("=> Mean Error (256): {:.6f}, AUC@0.07: {} based on maps".format(mean_error256 * 100., auc256))
    print("=> Mean Error (64): {:.6f}, AUC@0.07: {} based on maps".format(mean_error64 * 100., auc64))


