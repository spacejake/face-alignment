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

from utils.imutils import load_image, draw_labelmap, im_to_numpy, im_to_torch
from common import Split
from face_alignment.utils import shuffle_lr, flip, crop, getTransform, transform

'''
Modified derivative of https://github.com/hzh8311/pyhowfar
'''

class W300LP(data.Dataset):

    def __init__(self, args, split=Split.test):
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
        self.split = split
        self.is_train = True if self.split == Split.train else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
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
        inp, heatmap, pts, c, s = self.generateSampleFace(index)

        if self.is_train:
            return inp, heatmap, pts
        else:
            meta = {'index': index, 'center': c, 'scale': s, 'pts': pts,}
            return inp, heatmap, pts, meta

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        main_pts = sio.loadmat(
            os.path.join(self.lmk_dir, self.anno[idx].split('_')[0],
                         self.anno[idx][:-4] + '.mat'))
        pts = main_pts['pts_2d'] if self.pointType == '2D' else main_pts['pts_3d']
        pts = torch.from_numpy(pts)
        c = torch.Tensor((450 / 2, 450 / 2 + 50))
        s = 1.8

        img = load_image(
            os.path.join(self.img_dir, self.anno[idx].split('_')[0], self.anno[idx][:-8] +
                         '.jpg'))

        # plt.imshow(im_to_numpy(img.clone()))
        # plt.show()


        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            if random.random() <= 0.5:
                img = flip(img).float()
                pts = shuffle_lr(pts, width=img.size(2))
                c[0] = img.size(2) - c[0]
                # plt.imshow(im_to_numpy(img))
                # plt.show()

            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

            # plt.imshow(im_to_numpy(img))
            # plt.show()

        inp = im_to_torch(crop(im_to_numpy(img), c, s, 256, rotate=r))
        # Transform Points
        ptsTransMat = getTransform(c, s, 256, rotate=r)
        for i in range(self.nParts):
            if pts[i, 0] > 0:
                pts[i] = transform(pts[i] + 1, ptsTransMat)


        # inp = color_normalize(inp, self.mean, self.std)

        # plt.imshow(im_to_numpy(inp))
        # plt.show()

        tpts = pts.clone()
        heatmap = torch.zeros(self.nParts, 64, 64)
        transMat = getTransform(c, s, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                # tpts[i, 0:2] = transform(tpts[i, 0:2] + 1, transMat)
                heatmap[i] = draw_labelmap(heatmap[i], tpts[i] - 1, sigma=1)

        return inp, heatmap, pts, c, s

    def _comput_mean(self):
        meanstd_file = './data/300W_LP/mean.pth.tar'
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
    import utils.opts as opts
    args = opts.argparser()

    # dataset = W300LP(args, Split.test)
    dataset = W300LP(args, Split.train)
    crop_win = None
    for i in range(dataset.__len__()):
        input, target, tpts = dataset.__getitem__(i)
        input = im_to_numpy(input)
        target = target.numpy()
        pts = tpts.numpy()
        # if crop_win is None:
        #     crop_win = plt.imshow(input)
        # else:
        #     crop_win.set_data(input)
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input)
        ax.plot(pts[0:17, 0], pts[0:17, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[17:22, 0], pts[17:22, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[22:27, 0], pts[22:27, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[27:31, 0], pts[27:31, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[31:36, 0], pts[31:36, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[36:42, 0], pts[36:42, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[42:48, 0], pts[42:48, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[48:60, 0], pts[48:60, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.plot(pts[60:68, 0], pts[60:68, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
        ax.axis('off')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(pts[:, 0] * 1.2, pts[:, 1], pts[:, 2], c="cyan", alpha=1.0, edgecolor='b')
        ax.plot3D(pts[:17, 0] * 1.2, pts[:17, 1], pts[:17, 2], color='blue')
        ax.plot3D(pts[17:22, 0] * 1.2, pts[17:22, 1], pts[17:22, 2], color='blue')
        ax.plot3D(pts[22:27, 0] * 1.2, pts[22:27, 1], pts[22:27, 2], color='blue')
        ax.plot3D(pts[27:31, 0] * 1.2, pts[27:31, 1], pts[27:31, 2], color='blue')
        ax.plot3D(pts[31:36, 0] * 1.2, pts[31:36, 1], pts[31:36, 2], color='blue')
        ax.plot3D(pts[36:42, 0] * 1.2, pts[36:42, 1], pts[36:42, 2], color='blue')
        ax.plot3D(pts[42:48, 0] * 1.2, pts[42:48, 1], pts[42:48, 2], color='blue')
        ax.plot3D(pts[48:, 0] * 1.2, pts[48:, 1], pts[48:, 2], color='blue')

        ax.view_init(elev=90., azim=90., )
        ax.set_xlim(ax.get_xlim()[::-1])

        plt.pause(0.5)
        plt.draw