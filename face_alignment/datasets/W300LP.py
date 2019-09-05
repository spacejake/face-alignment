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

from face_alignment.datasets.common import Split, Target, compute_laplacian, SpatialSoftmax
from face_alignment.utils import shuffle_lr, flip, crop, getTransform, transform, draw_gaussian, get_preds_fromhm, draw_gaussianv2
from face_alignment.util.imutils import *
from face_alignment.util.evaluation import get_preds
from face_alignment.util.heatmap import make_gauss, heatmaps_to_coords
from face_alignment.util.evaluation import get_preds, accuracy_points, calc_metrics

'''
Modified derivative of https://github.com/hzh8311/pyhowfar
'''

class W300LP(data.Dataset):

    def __init__(self, args, split='train', demo=False, mixcut=True):
        self.nParts = 68
        self.pointType = args.pointType
        self.img_dir = args.data
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor
        self.demo = demo
        self.mixcut = mixcut

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

        # Load pre-computed laplacian matrix
        laplacianData = sio.loadmat(
            os.path.join(self.img_dir, 'laplacian.mat'))
        self.laplcian = torch.from_numpy(laplacianData['L']).float()

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

    def _load_img(self, index):
        return load_image(os.path.join(self.img_dir, self.anno[index].split('_')[0],
                                       self.anno[index][:-4] + '.jpg'))

    def _load_anno(self, index):
        main_pts = sio.loadmat(
            os.path.join(self.lmk_dir, self.anno[index].split('_')[0],
                         self.anno[index][:-4] + '.mat'))
        orig_pts = main_pts['pts_2d'] if self.pointType == '2D' else main_pts['pts_3d']
        orig_pts = torch.from_numpy(orig_pts)
        return orig_pts

    def __getitem__(self, index):
        inp, heatmap64, heatmap256, pts, pts64, lap_pts, center, scale = self.generateSampleFace(index)
        target = Target(heatmap64, heatmap256, pts, pts64, lap_pts, center, scale)
        if self.is_train:
            return inp, target
        else:
            if not self.demo:
                meta = {'index': index, 'center': center, 'scale': scale} #, 'pts': pts,}
            else:
                # img_fn = os.path.join(self.anno[index].split('_')[0], self.anno[index][:-8] + '.jpg')
                img_fn = os.path.join(self.anno[index][:-4] + '.jpg')

                orig_img = self._load_img(index)
                orig_pts = self._load_anno(index)

                meta = {'index': index, 'center': center, 'scale': scale,\
                        'img_fn': img_fn, 'orig_img': orig_img, 'orig_pts': orig_pts}

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

        # if self.is_train and random.random() <= 0.4:
        #     eye_occlusion(img=img, center=raw_pts[27])
        # Mixcut
        # elif self.is_train and self.mixcut and random.random() <= 0.1:

        if self.is_train and self.mixcut and random.random() <= 0.5:
            rand_index = torch.randint(0, self.total, (1,))
            cut_img = load_image(
                os.path.join(self.img_dir, self.anno[rand_index].split('_')[0], self.anno[rand_index][:-8] +
                             '.jpg'))
            img = cutmix(img, cut_img)

        return self.genData(img, raw_pts, c, s, sf, rf)

    def genData(self, img, raw_pts, c, s, sf, rf):
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

        # show_joints(img, raw_pts)

        inp = im_to_torch(crop(im_to_numpy(img), c, s, 256, rotate=r))

        # # Perform cutout
        # if self.is_train and random.random() <= 0.5:
        #     inp = cutout(inp)

        # Transform Points
        # 256x256 GT Heatmap and Points
        pts = raw_pts.clone()
        heatmap256 = torch.zeros(self.nParts, 256, 256)
        transMat256 = getTransform(c, s, 256, rotate=r)
        for i in range(self.nParts):
            pts[i] = transform(pts[i], transMat256)
            if pts[i, 0] > 0:
                heatmap256[i] = make_gauss(pts[i, 0:2], (256, 256), sigma=2., normalize=True)

        # inp = color_normalize(inp, self.mean, self.std)

        # 64x64 Intermediate Heatmap
        pts64 = raw_pts.clone()
        heatmap64 = torch.zeros(self.nParts, 64, 64)
        transMat64 = getTransform(c, s, 64, rotate=r)
        for i in range(self.nParts):
            pts64[i] = transform(pts64[i], transMat64)
            if pts64[i, 0] > 0:
                heatmap64[i] = make_gauss(pts64[i, 0:2], (64, 64), sigma=1., normalize=True)

        # Compute Target Laplacian vectors
        if self.laplcian is not None:
            lap_pts = compute_laplacian(self.laplcian, pts)
        else:
            # Not used during test in some datasets for now
            lap_pts = pts.clone()

        return inp, heatmap64, heatmap256, pts, pts64, lap_pts, c, s

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
                                            self.anno[i][:-4] + '.jpg')
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


if __name__=="__main__":
    from face_alignment.datasets.common import SpatialSoftmax
    import face_alignment.util.opts as opts

    args = opts.argparser()
    # args.data = "../../data/300W_LP"
    # dataset = W300LP(args, Split.test)
    datasetLoader = W300LP
    crop_win = None
    loader = torch.utils.data.DataLoader(
        datasetLoader(args, 'test'),
        batch_size=args.val_batch,
        #shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    idx = range(1, 69, 1)
    # layer = SpatialSoftmax(256, 256, 68, temperature=1., unnorm=True)

    all_dists256 = torch.zeros((68, loader.dataset.__len__()))
    all_dists64 = torch.zeros((68, loader.dataset.__len__()))
    for val_idx, data in enumerate(loader):
        input, label, meta = data
        # input, label = data
        target = Target._make(label)

        # rand_index = torch.randperm(input.size()[0]).cuda()


        npimg = im_to_numpy(input[0])
        # io.imsave("eye_occlude.png", npimg)

        # show_joints3D(target.pts.squeeze(0))
        # show_joints(input.squeeze(0), target.pts.squeeze(0))
        # show_heatmap(target.heatmap64)
        # show_heatmap(target.heatmap256)

        # test_hmpred = heatmaps_to_coords(target.heatmap256)
        #
        test_hmpred = heatmaps_to_coords(target.heatmap256)

        acc256, batch_dists256 = accuracy_points(test_hmpred, target.pts[:,:,:2], idx, thr=0.07)
        all_dists256[:, val_idx * args.val_batch:(val_idx + 1) * args.val_batch] = batch_dists256

        show_joints(input.squeeze(0), test_hmpred.squeeze(0))
        #
        # # sample_hm = sample_with_heatmap(input.squeeze(0), target.heatmap64.squeeze(0))
        # # plt.imshow(sample_hm)

        # # TEST 64 heatmap extraction
        test_hmpred = heatmaps_to_coords(target.heatmap64)
        test_hmpred = test_hmpred * 4 # 64->256
        acc64, batch_dists64 = accuracy_points(test_hmpred, target.pts[:,:,:2], idx, thr=0.07)
        all_dists64[:, val_idx * args.val_batch:(val_idx + 1) * args.val_batch] = batch_dists64

        show_joints(input.squeeze(0), test_hmpred.squeeze(0))

        plt.pause(0.5)
        plt.draw()

    mean_error256 = torch.mean(all_dists256)
    mean_error64 = torch.mean(all_dists64)

    auc256 = calc_metrics(all_dists256, path=args.checkpoint, category='300W-Testset-256',
                          method='SoftArgmax 256')  # this is auc of predicted maps and target.
    auc64 = calc_metrics(all_dists64, path=args.checkpoint, category='300W-Testset-64',
                         method='SoftArgmax 64', line='g-')  # this is auc of predicted maps and target.

    print("=> Mean Error (256): {}, AUC@0.07: {} based on maps".format(mean_error256 * 100., auc256))
    print("=> Mean Error (64): {}, AUC@0.07: {} based on maps".format(mean_error64 * 100., auc64))

