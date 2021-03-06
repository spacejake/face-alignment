import os
import time
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
from progress.bar import Bar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.model_zoo import load_url
import itertools

from face_alignment import models
from face_alignment.api import NetworkSize
from face_alignment.models import FAN, ResNetDepth
from face_alignment.utils import *

from face_alignment.datasets.W300LP import W300LP
from face_alignment.datasets.AFLW2000 import AFLW2000
from face_alignment.datasets.common import Target


from face_alignment.util.logger import Logger, savefig
from face_alignment.util.imutils import show_joints3D, show_heatmap, sample_with_heatmap
from face_alignment.util.evaluation import AverageMeter, calc_metrics, accuracy_points, get_preds, accuracy_depth
from face_alignment.util.misc import adjust_learning_rate, save_checkpoint, save_pred
import face_alignment.util.opts as opts

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

best_acc = 0.
best_auc = 0.
idx = range(1, 69, 1)
gauss_256 = None

class Model(NamedTuple):
    FAN: torch.nn.Module
    Depth: torch.nn.Module

    def eval(self):
        if self.FAN is not None:
            self.FAN.eval()
        if self.Depth is not None:
            self.Depth.eval()

    def train(self):
        if self.FAN is not None:
            self.FAN.train()
        if self.Depth is not None:
            self.Depth.train()



class Criterion(NamedTuple):
    hm: torch.nn.Module
    pts: torch.nn.Module

class Optimizer(NamedTuple):
    FAN: torch.optim.Optimizer
    Depth: torch.optim.Optimizer

def get_loader(data):
    dataset = os.path.basename(os.path.normpath(data))
    return {
        '300W_LP': W300LP,
        # 'LS3D-W/300VW-3D': VW300,
        'AFLW2000': AFLW2000,
        # 'LS3D-W': LS3DW,
    }[dataset]

models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}

def main(args):
    global best_acc
    global best_auc
    global gauss_256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    train_fan = args.train_fan
    train_depth = args.train_depth


    # If both or none are specified, train both
    if train_fan == train_depth:
        train_fan = True
        train_depth = True

    print("==> Creating model '{}-{}', stacks={}, blocks={}, feats={}".format(
        args.netType, args.pointType, args.nStacks, args.nModules, args.nFeats))

    print("=> Models will be saved at: {}".format(args.checkpoint))

    # Network Models
    network_size = int(args.nStacks)
    if train_fan:
        face_alignment_net = FAN(network_size)
    else:
        print("Training only Depth...")
        face_alignment_net = None

    if train_depth:
        depth_net = ResNetDepth()
    else:
        print("Training only FAN...")
        depth_net = None

    if torch.cuda.device_count() > 1:
        deviceList = None
        nGpus = torch.cuda.device_count()
        if (args.devices is not None):
            deviceList = args.devices
            nGpus = len(deviceList)
        # elif args.nGpu > 1:
        #     nGpus = args.nGpu

        print("Using ", nGpus, "GPUs({})...".format(deviceList))
        if train_fan:
            face_alignment_net = torch.nn.DataParallel(face_alignment_net, device_ids=deviceList)

        if train_depth:
            depth_net = torch.nn.DataParallel(depth_net, device_ids=deviceList)

    if train_fan:
        face_alignment_net = face_alignment_net.to(device)

    if train_depth:
        depth_net = depth_net.to(device)

    model = Model(face_alignment_net, depth_net)

    # Loss Functions
    hm_crit = torch.nn.MSELoss(reduction='mean').to(device)
    pnt_crit = torch.nn.MSELoss(reduction='mean').to(device)
    criterion = Criterion(hm_crit, pnt_crit)

    # Optimization
    if train_fan:
        optimizerFan = torch.optim.RMSprop(
            model.FAN.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizerFan = None

    if train_depth:
        optimizerDepth = torch.optim.RMSprop(
            model.Depth.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizerDepth = None

    optimizer = Optimizer(optimizerFan, optimizerDepth)

    # Load Data
    title = args.checkpoint.split('/')[-1] + ' on ' + args.data.split('/')[-1]
    Loader = get_loader(args.data)

    val_loader = torch.utils.data.DataLoader(
        Loader(args, 'test'),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.resume and train_fan:
        if os.path.isfile(args.resume):
            print("=> Loading FAN checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            # model.FAN.load_state_dict(checkpoint['state_dict'])
            fan_weights = {
                k.replace('module.', ''): v for k,
                v in checkpoint['state_dict'].items()}
            model.FAN.load_state_dict(fan_weights)

            optimizer.FAN.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded FAN checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger_fn = os.path.join(args.checkpoint, 'log.txt')
            logger = Logger(logger_fn, title=title, resume=os.path.isfile(logger_fn))
        else:
            print("=> no FAN checkpoint found at '{}'".format(args.resume))
    else:
        if args.pretrained and train_fan:
            fan_weights = load_url(models_urls['3DFAN-4'], map_location=lambda storage, loc: storage)
            model.FAN.load_state_dict(fan_weights)

        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Train 3D Loss', 'Valid Loss', 'Valid 3D Loss', 'Train Acc', 'Val Acc', 'AUC'])

    if args.resume_depth and train_depth:
        if os.path.isfile(args.resume_depth):
            print("=> Loading Depth checkpoint '{}'".format(args.resume_depth))
            checkpoint = torch.load(args.resume_depth)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            # model.Depth.load_state_dict(checkpoint['state_dict'])
            depth_weights = {
                k.replace('module.', ''): v for k,
                v in checkpoint['state_dict'].items()}
            model.Depth.load_state_dict(depth_weights)

            optimizer.Depth.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded Depth checkpoint '{}' (epoch {})".format(args.resume_depth, checkpoint['epoch']))
        else:
            print("=> no Depth checkpoint found at '{}'".format(args.resume_depth))
    elif args.pretrained and train_depth:
        depth_weights = load_url(models_urls['depth'], map_location=lambda storage, loc: storage)
        depth_dict = {
            k.replace('module.', ''): v for k,
                                            v in depth_weights['state_dict'].items()}
        model.Depth.load_state_dict(depth_dict)

    cudnn.benchmark = True
    if train_fan:
        print('=> Total params: %.2fM' % (sum(p.numel() for p in model.FAN.parameters()) / (1024. * 1024)))

    if args.evaluation:
        print('=> Evaluation only')
        D = args.data.split('/')[-1]
        save_dir = os.path.join(args.checkpoint, D)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        loss, loss_depth, acc, predictions, auc = validate(val_loader, model, criterion, args.netType,
                                                        args.debug, args.flip, device=device)
        save_pred(predictions, checkpoint=save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Loader(args, split='train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    lr = args.lr

    for epoch in range(args.start_epoch, args.epochs):
        if optimizer.FAN is not None:
            lr_fan = adjust_learning_rate(optimizer.FAN, epoch, lr, args.schedule, args.gamma)

        if optimizer.Depth is not None:
            lr_depth = adjust_learning_rate(optimizer.Depth, epoch, lr, args.schedule, args.gamma)

        # New Learning rate
        lr = lr_fan if optimizer.FAN is not None else lr_depth

        print('=> Epoch: %d | LR %.8f' % (epoch + 1, lr))

        train_loss, train_losslmk, train_acc = train(train_loader, model, criterion, optimizer, args.netType, epoch,
                                      debug=args.debug, flip=args.flip, device=device)

        # do not save predictions in model file
        valid_loss, valid_losslmk, valid_acc, predictions, valid_auc = validate(val_loader, model, criterion, args.netType,
                                                      args.debug, args.flip, device=device)

        logger.append([int(epoch + 1), lr, train_loss, train_losslmk, valid_loss, valid_losslmk, train_acc, valid_acc, valid_auc])

        is_best = valid_auc >= best_auc
        best_auc = max(valid_auc, best_auc)

        if train_fan:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'netType': args.netType,
                    'state_dict': model.FAN.state_dict(),
                    'best_acc': best_auc,
                    'optimizer': optimizer.FAN.state_dict(),
                },
                is_best,
                predictions,
                checkpoint=args.checkpoint,
                filename="checkpointFAN.pth.tar")

        if train_depth:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'iter': 0,
                    'netType': args.netType,
                    'state_dict': model.Depth.state_dict(),
                    'best_acc': best_auc,
                    'optimizer': optimizer.Depth.state_dict(),
                },
                is_best,
                None,
                checkpoint=args.checkpoint,
                filename="checkpointDepth.pth.tar")
        
        savefig(os.path.join(args.checkpoint, 'log_iter.eps'))

    logger.close()
    logger.plot(['AUC'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(loader, model, criterion, optimizer, netType, epoch, iter=0, debug=False, flip=False, device='cuda:0'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losseslmk = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    train_fan = model.FAN is not None
    train_depth = model.Depth is not None

    # rnn = torch.nn.LSTM(10, 20, 2)
    # hidden = torch.autograd.Variable(torch.zeros((args.train_batch)))

    gt_win, pred_win = None, None
    bar = Bar('Training', max=len(loader))
    for loader_idx, (inputs, label) in enumerate(loader):

        target = Target._make(label)
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.to(device))
        target_hm64 = torch.autograd.Variable(target.heatmap64.to(device))
        target_pts = torch.autograd.Variable(target.pts.to(device))

        # FAN
        loss = torch.zeros([1], dtype=torch.float32)[0]
        if train_fan:
            # Forward
            output = model.FAN(input_var)
            out_hm = output[-1]

            if flip:
                flip_output = model.FAN(flip(out_hm[-1].clone()), is_label=True)
                out_hm += flip(flip_output[-1])

            out_hm = out_hm.cpu()

            # Supervision
            # Intermediate supervision
            loss = 0
            for out_inter in output:
                loss += criterion.hm(out_inter, target_hm64)

            # Back-prop
            optimizer.FAN.zero_grad()
            loss.backward()
            optimizer.FAN.step()
        else:
            out_hm = target.heatmap64

        if train_fan:
            pts, _ = get_preds_fromhm(out_hm.detach().cpu(), target.center, target.scale)
            pts = pts * 4 # 64->256
        else:
            pts = target.pts[:,:,:2]

        # DEPTH
        lossDepth =  torch.zeros([1], dtype=torch.float32)[0]
        if train_depth:
            target_hm256 = torch.autograd.Variable(target.heatmap256.to(device))
            depth_inp = torch.cat((input_var, target_hm256), 1)
            depth_pred = model.Depth(depth_inp)

            # Supervision
            # 3D LMk Loss
            lossDepth = criterion.pts(depth_pred, target_pts[:,:,2])
            depth_pred = depth_pred.cpu()

            # Back-prop
            optimizer.Depth.zero_grad()
            lossDepth.backward()
            optimizer.Depth.step()

            pts_img = torch.cat((pts, depth_pred.unsqueeze(2)), 2)
        else:
            pts_img = torch.cat((pts, target.pts[:,:,2].unsqueeze(2)), 2)

        if train_fan and train_depth:
            acc, _ = accuracy_points(pts_img, target.pts, idx, thr=0.07)
        elif train_fan:
            acc, _ = accuracy_points(pts_img[...,:2], target.pts[...,:2], idx, thr=0.07)
        elif train_depth:
            acc, _ = accuracy_depth(pts_img[...,2:], target.pts[...,2:], idx, thr=0.07)

        # acc, _ = accuracy_points(pts_img, target.pts, idx, thr=0.07)

        losses.update(loss.data, inputs.size(0))
        losseslmk.update(lossDepth.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        if loader_idx % 50 == 0:
            show_joints3D(pts_img.detach()[0], outfn=os.path.join(args.checkpoint,"3dPoints.png"))
            show_joints3D(target.pts[0], outfn=os.path.join(args.checkpoint,"3dPoints_gt.png"))

            show_heatmap(out_hm.cpu().data[0].unsqueeze(0), outname=os.path.join(args.checkpoint, "hm64.png"))
            show_heatmap(target.heatmap64.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint, "hm64_gt.png"))

            sample_hm = sample_with_heatmap(inputs[0], out_hm[0].detach())
            io.imsave(os.path.join(args.checkpoint,"input-with-hm64.png"),sample_hm)
            sample_hm = sample_with_heatmap(inputs[0], target.heatmap64[0])
            io.imsave(os.path.join(args.checkpoint,"input-with-gt-hm64.png"),sample_hm)

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | LossLmk: {losslmk: .4f} | Acc: {acc: .4f}'.format(
            batch=loader_idx + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            losslmk=losseslmk.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    return losses.avg, losseslmk.avg, acces.avg

def validate(loader, model, criterion, netType, debug, flip, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losseslmk = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    predictions = torch.Tensor(loader.dataset.__len__(), 68, 3)

    model.eval()

    val_fan = model.FAN is not None
    val_depth = model.Depth is not None

    gt_win, pred_win = None, None
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((68, loader.dataset.__len__()))
    gauss_256 = None
    for val_idx, (inputs, label, meta) in enumerate(loader):
        target = Target._make(label)
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.to(device))
        target_var = target.heatmap64.to(device)
        target_pts = target.pts.to(device)

        loss = torch.zeros([1], dtype=torch.float32)[0]
        if val_fan:
            output = model.FAN(input_var)
            out_hm = output[-1]

            if flip:
                flip_output = model.FAN(flip(out_hm[-1].detach()))
                out_hm += flip(flip_output[-1], is_label=True)

            out_hm = out_hm.cpu()

            for o in output:
                loss += criterion.hm(o, target_var)
        else:
            out_hm = target.heatmap64

        if val_fan:
            pts, _ = get_preds_fromhm(out_hm.cpu(), target.center, target.scale)
            pts = pts * 4 # 64->256
            heatmaps, gauss_256 = gen_heatmap(pts, dim=(pts.size(0), 68, 256, 256), sigma=2, g=gauss_256)
            heatmaps = heatmaps.to(device)
        else:
            pts = target.pts[:,:,:2]
            heatmaps = target.heatmap256.to(device)

        lossDepth = torch.zeros([1], dtype=torch.float32)[0]
        if val_depth:
            depth_inp = torch.cat((input_var, heatmaps), 1)
            depth_pred = model.Depth(depth_inp).detach()

            # intermediate supervision
            lossDepth = criterion.pts(depth_pred, target_pts[:,:,2])

            depth_pred = depth_pred.cpu()
            pts_img = torch.cat((pts.data, depth_pred.detach().data.unsqueeze(2)), 2)
        else:
            pts_img = torch.cat((pts.data, target.pts[:,:,2].unsqueeze(2)), 2)

        if val_idx % 50 == 0:
            show_joints3D(pts_img.detach()[0], outfn=os.path.join(args.checkpoint,"val_3dPoints.png"))
            show_joints3D(target.pts[0], outfn=os.path.join(args.checkpoint,"val_3dPoints_gt.png"))
            show_heatmap(out_hm.data[0].cpu().unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm64.png"))
            show_heatmap(target.heatmap64.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm64_gt.png"))
            show_heatmap(heatmaps.data[0].cpu().unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm256.png"))
            show_heatmap(target.heatmap256.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm256_gt.png"))
            sample_hm = sample_with_heatmap(inputs[0], out_hm[0].detach())
            io.imsave(os.path.join(args.checkpoint,"val_input-with-hm64.png"),sample_hm)
            sample_hm = sample_with_heatmap(inputs[0], target.heatmap64[0])
            io.imsave(os.path.join(args.checkpoint,"val_input-with-gt-hm64.png"),sample_hm)

        if val_fan and val_depth:
            acc, batch_dists = accuracy_points(pts_img, target.pts, idx, thr=0.07)
        elif val_fan:
            acc, batch_dists = accuracy_points(pts_img[...,:2], target.pts[...,:2], idx, thr=0.07)
        elif val_depth:
            acc, batch_dists = accuracy_depth(pts_img[...,2:], target.pts[...,2:], idx, thr=0.07)

        # acc = (acc2D + accZ)/2
        # batch_dists = batch_dists2D + batch_distsZ

        all_dists[:, val_idx * args.val_batch:(val_idx + 1) * args.val_batch] = batch_dists

        for n in range(inputs.size(0)):
            predictions[meta['index'][n], :, :] = pts_img[n, :, :]

        losses.update(loss.data, inputs.size(0))
        losseslmk.update(lossDepth.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | LossLmk: {losslmk:.4f} | Acc: {acc: .4f}'.format(
            batch=val_idx + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            losslmk=losseslmk.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists, path=args.checkpoint, category='300W-Testset', method='3D-FAN') # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))
    return losses.avg, losseslmk.avg, acces.avg, predictions, auc


if __name__ == '__main__':
    args = opts.argparser()

    main(args)
