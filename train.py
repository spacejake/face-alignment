import os
import time
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
from progress.bar import Bar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import itertools

from face_alignment import models
from face_alignment.api import NetworkSize
from face_alignment.models import FAN, ResNetDepth
from face_alignment.utils import *

from datasets.W300LP import W300LP
from datasets.common import Split, Target


from utils.logger import Logger, savefig
from utils.imutils import batch_with_heatmap, show_joints3D, show_heatmap
from utils.evaluation import accuracy, AverageMeter, calc_metrics, calc_dists, accuracy_points
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred
import opts

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

best_acc = 0.
best_auc = 0.
idx = range(1, 69, 1)

class Model(NamedTuple):
    FAN: torch.nn.Module
    Depth: torch.nn.Module

    def eval(self):
        self.FAN.eval()
        self.Depth.eval()

    def train(self):
        self.FAN.train()
        self.Depth.train()


class Criterion(NamedTuple):
    hm: torch.nn.Module
    pts: torch.nn.Module

def get_loader(data):
    dataset = os.path.basename(os.path.normpath(data))
    return {
        '300W_LP': W300LP,
        # 'LS3D-W/300VW-3D': VW300,
        # 'AFLW2000': AFLW2000,
        # 'LS3D-W': LS3DW,
    }[dataset]

def main(args):
    global best_acc
    global best_auc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    print("==> Creating model '{}-{}', stacks={}, blocks={}, feats={}".format(
        args.netType, args.pointType, args.nStacks, args.nModules, args.nFeats))

    print("=> Models will be saved at: {}".format(args.checkpoint))

    # Network Models
    network_size = int(NetworkSize.LARGE)
    face_alignment_net = FAN(network_size)
    depth_net = ResNetDepth()

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs...")
        face_alignment_net = torch.nn.DataParallel(face_alignment_net)
        depth_net = torch.nn.DataParallel(depth_net)

    face_alignment_net = face_alignment_net.to(device)
    depth_net = depth_net.to(device)
    model = Model(face_alignment_net, depth_net)

    # Loss Functions
    hm_crit = torch.nn.MSELoss(size_average=True).to(device)
    pnt_crit = torch.nn.MSELoss(size_average=True).to(device)
    criterion = Criterion(hm_crit, pnt_crit)

    # Optimization
    optimizer = torch.optim.RMSprop(
        itertools.chain(model.FAN.parameters(), model.Depth.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load Data
    title = args.checkpoint.split('/')[-1] + ' on ' + args.data.split('/')[-1]
    Loader = get_loader(args.data)

    # val_loader = torch.utils.data.DataLoader(
    #     Loader(args, 'test'),
    #     batch_size=args.val_batch,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True)

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> Loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc = checkpoint['best_acc']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #         logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    # else:
    #     logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    #     logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])

    cudnn.benchmark = True
    # print('=> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / (1024. * 1024)))

    # if args.evaluation:
    #     print('=> Evaluation only')
    #     D = args.data.split('/')[-1]
    #     save_dir = os.path.join(args.checkpoint, D)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     loss, acc, predictions, auc = validate(val_loader, model, criterion, args.netType,
    #                                                     args.debug, args.flip, device=device)
    #     save_pred(predictions, checkpoint=save_dir)
    #     return

    train_loader = torch.utils.data.DataLoader(
        Loader(args, split='train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    lr = args.lr

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('=> Epoch: %d | LR %.8f' % (epoch + 1, lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.netType,
                                      args.debug, args.flip, device=device)

        # do not save predictions in model file
        # valid_loss, valid_acc, predictions, valid_auc = validate(val_loader, model, criterion, args.netType,
        #                                               args.debug, args.flip, device=device)
        #
        # logger.append([int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc])
        #
        # is_best = valid_auc >= best_auc
        # best_auc = max(valid_auc, best_auc)
        # save_checkpoint(
        #     {
        #         'epoch': epoch + 1,
        #         'netType': args.netType,
        #         'state_dict': model.FAN.state_dict(),
        #         'best_acc': best_auc,
        #         'optimizer': optimizer.state_dict(),
        #     },
        #     is_best,
        #     predictions,
        #     checkpoint=args.checkpoint)

    # logger.close()
    # logger.plot(['AUC'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(loader, model, criterion, optimizer, netType, debug=False, flip=False, device='cuda:0'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    # rnn = torch.nn.LSTM(10, 20, 2)
    # hidden = torch.autograd.Variable(torch.zeros((args.train_batch)))

    gt_win, pred_win = None, None
    bar = Bar('Training', max=len(loader))
    for loader_idx, (inputs, label) in enumerate(loader):

        target = Target._make(label)
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.to(device))
        target_hm = torch.autograd.Variable(target.heatmap.to(device))
        target_pts = torch.autograd.Variable(target.pts.to(device))

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            # pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                # plt.subplot(122)
                # pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                # pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        output = model.FAN(input_var)
        out_hm = output[-1]

        if flip:
            flip_output = model.FAN(flip(out_hm[-1].clone()), is_label=True)
            out_hm += flip(flip_output[-1])

        out_hm = out_hm.cpu()

        pts, pts_img = get_preds_fromhm(out_hm, target.center, target.scale)
        pts = pts * 4

        # Input for 3D Regressor
        # TODO: Problem, no back propagation to FAN Model, basically training them separately
        # if self.landmarks_type == LandmarksType._3D:
        heatmaps = np.zeros((pts.size(0), 68, 256, 256), dtype=np.float32)
        tpts = pts.clone()
        for b in range(pts.size(0)):
            for n in range(68):
                if tpts[b, n, 0] > 0:
                    heatmaps[b, n] = draw_gaussian(
                        heatmaps[b, n], tpts[b, n], 2)
        heatmaps = torch.from_numpy(heatmaps)
        heatmaps_var = torch.autograd.Variable(heatmaps.to(device))

        depth_inp = torch.cat((input_var, heatmaps_var), 1)
        depth_pred = model.Depth(depth_inp).data.cpu().unsqueeze(2)
        tscale = target.scale.unsqueeze(1).unsqueeze(2)
        depth_pred_scaled = depth_pred * (1.0 / (256.0 / (200.0 * tscale)))
        pts_img = torch.cat((pts_img, depth_pred_scaled), 2)

        # Intermediate supervision
        loss = 0
        for out_inter in output:
            loss += criterion.hm(out_inter, target_hm)

        # 3D LMk Loss
        loss += criterion.pts(pts_img.to(device), target_pts)

        acc, _ = accuracy_points(pts_img, target.pts, idx, thr=0.07)

        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=loader_idx + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()
        print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=loader_idx + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg))

        show_joints3D(pts_img.detach()[0])
        show_heatmap(out_hm.detach()[0].unsqueeze(0))
        plt.pause(0.5)
        plt.draw()
    bar.finish()

    return losses.avg, acces.avg

"""
def validate(loader, model, criterion, netType, debug, flip, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    predictions = torch.Tensor(loader.dataset.__len__(), 68, 2)

    model.eval()
    gt_win, pred_win = None, None
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((68, loader.dataset.__len__()))
    for i, (inputs, target, meta) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.to(device))
        target_var = torch.autograd.Variable(target.heatmap.to(device))
        target_pts = torch.autograd.Variable(target.pts.to(device))

        output = model(input_var)
        out_hm = output[-1]

        if flip:
            flip_output = model.FAN(flip(out_hm[-1].detach()), is_label=True)
            out_hm += flip(flip_output[-1])

        out_hm = out_hm.cpu()

        pts, pts_img = get_preds_fromhm(out_hm, target.center, target.scale)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

        # if self.landmarks_type == LandmarksType._3D:
        heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
        for i in range(68):
            if pts[i, 0] > 0:
                heatmaps[i] = draw_gaussian(
                    heatmaps[i], pts[i], 2)
        heatmaps = torch.from_numpy(
            heatmaps).unsqueeze_(0)

        heatmaps = heatmaps.to(device)
        depth_pred = model.Depth(
            torch.cat((input_var, heatmaps), 1)).data.cpu().view(68, 1)
        pts_img = torch.cat(
            (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * target.scale)))), 1)

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc, batch_dists = accuracy(score_map, target.cpu(), idx, thr=0.07)
        all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists) # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))
    return losses.avg, acces.avg, predictions, auc
"""


if __name__ == '__main__':
    args = opts.argparser()

    main(args)
