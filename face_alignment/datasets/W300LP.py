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

from face_alignment.datasets.common import Split, Target
from face_alignment.utils import shuffle_lr, flip, crop, getTransform, transform, draw_gaussian, get_preds_fromhm
from face_alignment.util.imutils import *
from face_alignment.util.evaluation import get_preds


'''
Modified derivative of https://github.com/hzh8311/pyhowfar
'''

def get_loader(data):
    dataset = os.path.basename(os.path.normpath(data))
    return {
        '300W_LP': W300LP,
        # 'LS3D-W/300VW-3D': VW300,
        # 'AFLW2000': AFLW2000,
        # 'LS3D-W': LS3DW,
    }[dataset]


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
        self.mean, self.std = self._preprocess()

        # Load pre-computed laplacian matrix
        laplacianData = sio.loadmat(
            os.path.join(self.img_dir, 'laplacian.mat'))
        self.laplcian = torch.from_numpy(laplacianData['L']).float()
        # Load faces, subtract by 1 to convert to 0-indexing in python (vs Matlab)
        self.faces = torch.from_numpy(laplacianData['F'])-1
        self.g64 = None
        self.g256 = None

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

        # Compute Target Laplacian vectors
        # lap_pts = compute_laplacian(self.laplcian, pts)

        #return inp, heatmap64, heatmap256, pts, lap_pts, c, s
        return inp, heatmap64, heatmap256, pts, c, s

    def _preprocess(self):
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

def compute_laplacian(laplacianMat, points):
    lap_pts = torch.matmul(laplacianMat, points)

    return lap_pts

if __name__=="__main__":
    import face_alignment.util.opts as opts

    args = opts.argparser()
    args.data = "../../"+args.data
    # dataset = W300LP(args, Split.test)
    datasetLoader = get_loader(args.data)
    crop_win = None
    loader = torch.utils.data.DataLoader(
        datasetLoader(args, 'train'),
        batch_size=1,
        #shuffle=True,
        num_workers=1,
        pin_memory=True)
    for i, data in enumerate(loader):
        input, label = data
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

        # Test other method
        test_hmpred = get_preds(target.heatmap64) * 4
        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        plt.pause(0.5)
        plt.draw()
