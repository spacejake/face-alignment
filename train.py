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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.model_zoo import load_url
import itertools

from face_alignment import models
from face_alignment.api import NetworkSize
from face_alignment.models import FAN, ResNetDepth, ResPatchDiscriminator, ResDiscriminator, NLayerDiscriminator
from face_alignment.utils import *

from face_alignment.datasets.W300LP import W300LP
from face_alignment.datasets.AFLW2000 import AFLW2000
from face_alignment.datasets.WFLW import WFLW
from face_alignment.datasets.common import Target, compute_laplacian, SpatialSoftmax, ConcatDataset


from face_alignment.util.logger import Logger, savefig
from face_alignment.util.imutils import show_joints3D, show_heatmap, sample_with_heatmap, im_to_numpy
from face_alignment.util.evaluation import AverageMeter, calc_metrics, accuracy_points, get_preds, accuracy_depth
from face_alignment.util.misc import adjust_learning_rate, save_checkpoint, save_pred
from face_alignment.util.heatmap import js_reg_losses, js_loss, euclidean_losses, average_loss, hm_losses
import face_alignment.util.opts as opts
from face_alignment.util.heatmap import make_gauss, heatmaps_to_coords

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

best_acc = 0.
best_auc = 0.
idx = range(1, 69, 1)
gauss_256 = None

class Model(NamedTuple):
    FAN: torch.nn.Module
    D_hm: torch.nn.Module
    Depth: torch.nn.Module

    def eval(self):
        if self.FAN is not None:
            self.FAN.eval()
            self.D_hm.eval()

        if self.Depth is not None:
            self.Depth.eval()

    def train(self):
        if self.FAN is not None:
            self.FAN.train()
            self.D_hm.eval()

        if self.Depth is not None:
            self.Depth.train()



class Criterion(NamedTuple):
    hm: torch.nn.Module
    d_hm: torch.nn.Module
    pts: torch.nn.Module
    laplacian: torch.nn.Module

class Optimizer(NamedTuple):
    FAN: torch.optim.Optimizer
    D_hm: torch.optim.Optimizer
    Depth: torch.optim.Optimizer

def get_loader(data):
    dataset = os.path.basename(os.path.normpath(data))
    return {
        '300W_LP': W300LP,
        # 'LS3D-W/300VW-3D': VW300,
        'AFLW2000': AFLW2000,
        # 'LS3D-W': LS3DW,
        'WFLW': WFLW,
    }[dataset]

def get_agan_threshold(a, iter, min):
    return a*iter**2+min

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
    network_size = args.nStacks
    if train_fan:
        face_alignment_net = FAN(num_modules=network_size)

        # fan_D = ResPatchDiscriminator(in_channels=71, ndf=8, ndlayers=2, use_sigmoid=True) #3-ch image + 68-ch heatmap
        fan_D = ResDiscriminator(in_channels=15, ndf=96, ndlayers=4, use_sigmoid=True)  # 3-ch image + 12-ch heatmap
        # fan_D = NLayerDiscriminator(input_nc=71, ndf=16, n_layers=1, use_sigmoid=True)  # 3-ch image + 68-ch heatmap
    else:
        print("Training only Depth...")
        face_alignment_net = None
        fan_D = None

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
            fan_D = torch.nn.DataParallel(fan_D, device_ids=deviceList)

        if train_depth:
            depth_net = torch.nn.DataParallel(depth_net, device_ids=deviceList)

    if train_fan:
        face_alignment_net = face_alignment_net.to(device)
        fan_D = fan_D.to(device)

    if train_depth:
        depth_net = depth_net.to(device)

    model = Model(face_alignment_net, fan_D, depth_net)

    # Loss Functions
    hm_crit = torch.nn.MSELoss(reduction='mean').to(device)
    crit_gan = torch.nn.MSELoss(reduction='mean').to(device)
    pnt_crit = torch.nn.MSELoss(reduction='mean').to(device)
    lap_crit = torch.nn.MSELoss(reduction='mean').to(device)
    criterion = Criterion(hm_crit, crit_gan, pnt_crit, lap_crit)

    # Optimization
    lr_hm_d = args.lr#*0.1
    if train_fan:
        print("Heatmap Discriminator initial lr: {}".format(lr_hm_d))
        optimizerFan = torch.optim.RMSprop(
            model.FAN.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        optimizerFanD = torch.optim.Adam(
            model.D_hm.parameters(),
            lr=lr_hm_d, weight_decay=args.weight_decay)
    else:
        optimizerFan = None
        optimizerFanD = None

    if train_depth:
        optimizerDepth = torch.optim.RMSprop(
            model.Depth.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizerDepth = None

    optimizer = Optimizer(optimizerFan, optimizerFanD, optimizerDepth)

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
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Train LossD', 'Train 3D Loss', 'Train Depth Loss', 'Train Laplacian Loss', 'Valid Loss', 'Valid 3D Loss', 'Train Acc', 'Val Acc', 'AUC'])

    if args.resume_gan and train_fan:
        if os.path.isfile(args.resume_gan):
            print("=> Loading Discriminator checkpoint '{}'".format(args.resume_gan))
            checkpoint = torch.load(args.resume_gan)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.D_hm.load_state_dict(checkpoint['state_dict'])
            optimizer.D_hm.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded Discriminator checkpoint '{}' (epoch {})".format(args.resume_gan, checkpoint['epoch']))
            # logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no Discriminator checkpoint found at '{}'".format(args.resume_gan))

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

    train_dataset = Loader(args, split='train')
    EyeLoader = get_loader(args.data_eyes)
    eye_args = args
    eye_args.data = args.data_eyes
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset([train_dataset, EyeLoader(eye_args, split='train')]),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True)

    lr = args.lr

    ## A-GAN (Adaptive GAN Thresholding)
    agan_error_threshold_decrease = 0.001 # Make Threshold reduction step wnen error below this value

    agan_threshold_coeff = 5e-4
    agan_threshold_max_steps = 13
    agan_threshold_min = 0.02

    # Initialize
    agan_threshold_step = agan_threshold_max_steps
    agan_threshold = get_agan_threshold(agan_threshold_coeff,
                                          agan_threshold_step,
                                          agan_threshold_min)
    print("Conditional GAN Initial threshold: {}".format(agan_threshold))

    ## Train
    for epoch in range(args.start_epoch, args.epochs):
        if optimizer.FAN is not None:
            lr_fan = adjust_learning_rate(optimizer.FAN, epoch, lr, args.schedule, args.gamma)
            lr_hm_d = adjust_learning_rate(optimizer.D_hm, epoch, lr_hm_d, args.schedule, args.gamma)

        if optimizer.Depth is not None:
            lr_depth = adjust_learning_rate(optimizer.Depth, epoch, lr, args.schedule, args.gamma)

        # New Learning rate
        lr = lr_fan if optimizer.FAN is not None else lr_depth
        print('=> Epoch: %d | LR_G %.8f | LR_D %.8f' % (epoch + 1, lr, lr_hm_d))

        train_loss, loss_d, train_lossreg, train_lossdepth, train_losslap, train_acc = \
            train(train_loader, model, criterion, optimizer, args.netType, epoch, train_dataset.laplcian,
                  debug=args.debug, flip=args.flip, device=device, conf_gan_thr=agan_threshold)

        # do not save predictions in model file
        valid_loss, valid_losslmk, valid_acc, predictions, valid_auc = validate(val_loader, model, criterion, args.netType,
                                                      args.debug, args.flip, device=device)

        logger.append([int(epoch + 1), lr, train_loss, loss_d, train_lossreg, train_lossdepth, train_losslap, valid_loss, valid_losslmk, train_acc, valid_acc, valid_auc])

        is_best = valid_auc >= best_auc
        best_auc = max(valid_auc, best_auc)

        # Slowly increase the Descriminator's allowed threshold to increase the challenge of defeating the network
        if train_loss < agan_error_threshold_decrease and agan_threshold_step > 0:
            agan_threshold_step -= 1
            agan_threshold = get_agan_threshold(agan_threshold_coeff,
                                                  agan_threshold_step,
                                                  agan_threshold_min)
            print("Gan error threshold met: {} < {}, decreasing Descriminator's threshold: {}".format(loss_g, agan_error_threshold_decrease, agan_threshold))


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

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'netType': args.netType,
                    'state_dict': model.D_hm.state_dict(),
                    'best_acc': best_auc,
                    'optimizer': optimizer.D_hm.state_dict(),
                },
                is_best,
                None,
                checkpoint=args.checkpoint,
                filename="checkpointGAN.pth.tar")

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


def backwardG(fake, loss_hm, model, opt, crit, weight_hm=1.0):

    # GAN Loss
    pred_fake = model(fake)
    true = torch.ones(fake.shape[0]).cuda()
    loss_G = crit(pred_fake, true)

    # Combined Loss
    loss_G_total = loss_G +  weight_hm * loss_hm

    # backward
    opt.zero_grad()
    loss_G_total.backward()
    opt.step()

    return loss_G_total, loss_G


def backwardD(fake, real, model, opt, crit):

    # Fake Score is %NME of points < threshold per batch
    fake_score = torch.zeros(fake.shape[0]).cuda()
    real_score = torch.ones(fake.shape[0]).cuda()

    # Train Real
    pred_real = model(real)
    loss_D_real = crit(pred_real, real_score)

    # Train Fake
    pred_fake = model(fake.detach())
    loss_D_fake = crit(pred_fake, fake_score)

    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    # backward
    opt.zero_grad()
    loss_D.backward()  # retain_graph=True)
    opt.step()

    return loss_D, loss_D_real, loss_D_fake


def train(loader, model, criterion, optimizer, netType, epoch, laplacian_mat,
          iter=0, debug=False, flip=False, device='cuda:0', conf_gan_thr=0.07):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesfan = AverageMeter()
    losses2d = AverageMeter()

    losses_d = AverageMeter()
    losses_d_real = AverageMeter()
    losses_d_fake = AverageMeter()

    lossesRegressor = AverageMeter()
    lossesDepth = AverageMeter()
    lossesLap = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    train_fan = model.FAN is not None
    train_depth = model.Depth is not None

    # rnn = torch.nn.LSTM(10, 20, 2)
    # hidden = torch.autograd.Variable(torch.zeros((args.train_batch)))

    gt_win, pred_win = None, None
    kl_loss = nn.KLDivLoss(size_average=False)
    bar = Bar('Training', max=len(loader))
    for loader_idx, (inputs, label) in enumerate(loader):
        batch_size = inputs.size(0)

        target = Target._make(label)
        data_time.update(time.time() - end)
        
        input_var = torch.autograd.Variable(inputs.to(device))
        target_hm64 = torch.autograd.Variable(target.heatmap64.to(device))
        target_hm_eye = torch.autograd.Variable(target.heatmap_eyes.to(device))
        target_pts64 = torch.autograd.Variable(target.pts64.to(device))
        target_pts = torch.autograd.Variable(target.pts.to(device))
        target_lap = torch.autograd.Variable(target.lap_pts.to(device))

        laplacian_mat = laplacian_mat.to(device)

        # FAN
        loss = torch.zeros([1], dtype=torch.float32)[0]
        loss_d = torch.zeros([1], dtype=torch.float32)[0]
        loss_d_real = torch.zeros([1], dtype=torch.float32)[0]
        loss_d_fake = torch.zeros([1], dtype=torch.float32)[0]
        if train_fan:
            # Forward
            out_hm, output = model.FAN(input_var)

            # Supervision
            # Intermediate supervision
            loss = 0
            # lossFan = 0
            # loss2D = 0

            target_lap64 = compute_laplacian(laplacian_mat, target_pts64)
            #Only supervise n-1 stacks with MSE
            for o in output[:-1]:
                loss += hm_losses(o, target_hm64)
                # Divergence Loss
                # loss += js_loss(o, target_hm64)

                # Point-to-point Loss
                #pts = heatmaps_to_coords(o)
                #loss += euclidean_losses(pts, target_pts64[:,:,:2]) #criterion.hm(pts, target_pts64)
                
            # Divergence and Point-to-point Loss
            loss += js_loss(out_hm, target_hm64)
            pts = heatmaps_to_coords(out_hm)
            loss += euclidean_losses(pts, target_pts64[:,:,:2]) #criterion.hm(pts, target_pts64)
            
            # Laplacian
            pred_pts64 = torch.cat((pts, target_pts64[:, :, 2:]), 2)
            pred_lap = compute_laplacian(laplacian_mat, pred_pts64)
            w_deform = 1.0 #1.5
            loss += w_deform * euclidean_losses(pred_lap, target_lap64)

            # scale 64->256
            pts = pts * 4

            loss = loss.mean()

            input_var64 = F.interpolate(input_var, size=(64,64), mode='bilinear', align_corners=False)

            if target.has_3d_anno.any().numpy() == 1:
                anno_3d_idx = target.has_3d_anno.squeeze().nonzero().squeeze(1)
                fake_in = torch.cat((input_var64[anno_3d_idx], out_hm[anno_3d_idx, 36:48]), 1)
                loss, loss_g = backwardG(fake_in, loss,
                                         model.D_hm, optimizer.FAN, criterion.d_hm,
                                         weight_hm=1.0)

            if target.has_2d_anno.any().numpy() == 1:
                anno_2d_idx = target.has_2d_anno.squeeze().nonzero().squeeze(1)
                fake_in = torch.cat((input_var64[anno_2d_idx], out_hm[anno_2d_idx, 36:48]), 1)
                real_in = torch.cat((input_var64[anno_2d_idx], target_hm64[anno_2d_idx, 36:48]), 1)
                loss_d, loss_d_real, loss_d_fake = backwardD(fake_in, real_in,
                                                             model.D_hm, optimizer.D_hm, criterion.d_hm)

            # Back-prop
            # optimizer.FAN.zero_grad()
            # loss.backward()
            # optimizer.FAN.step()



        else:
            pts = target_pts[:,:,:2]

        # DEPTH
        lossRegressor = torch.zeros([1], dtype=torch.float32)[0]
        lossDepth =  torch.zeros([1], dtype=torch.float32)[0]
        lossLap =  torch.zeros([1], dtype=torch.float32)[0]
        if train_depth:
            depth_inp = torch.cat((input_var, target_hm_eye), 1)
            depth_pred = model.Depth(depth_inp)

            # Supervision
            # Depth Loss
            lossDepth = euclidean_losses(depth_pred.unsqueeze(2), target_pts[:, :, 2:])

            # Laplacian Depth Loss
            # Computed for depth only, since both FAN and 3DRegressor are trained separably
            target_lap = torch.autograd.Variable(target.lap_pts.to(device))
            tpts256 = target_pts[:, :, 0:2]
            pred_pts256 = torch.cat((tpts256.to(device), depth_pred.unsqueeze(2)), 2)
            pred_lap = compute_laplacian(laplacian_mat.to(device), pred_pts256)
            lossLap = euclidean_losses(pred_lap, target_lap)

            lossRegressor = lossDepth + lossLap
            lossRegressor = lossRegressor.mean()

            # Back-prop
            optimizer.Depth.zero_grad()
            lossRegressor.backward()
            optimizer.Depth.step()

            pts_img = torch.cat((pts, depth_pred.unsqueeze(2)), 2).cpu()
        else:
            pts_img = torch.cat((pts, target_pts[:,:,2].unsqueeze(2)), 2).cpu()

        if train_fan and train_depth:
            acc, _ = accuracy_points(pts_img, target.pts, idx, thr=0.07)
        elif train_fan:
            acc, _ = accuracy_points(pts_img[...,:2], target.pts[...,:2], idx, thr=0.07)
        elif train_depth:
            acc, _ = accuracy_depth(pts_img[...,2:], target.pts[...,2:], idx, thr=0.07)

        # acc, _ = accuracy_points(pts_img, target.pts, idx, thr=0.07)

        losses.update(loss.data, batch_size)

        losses_d.update(loss_d.data, inputs.size(0))
        losses_d_real.update(loss_d_real.data, inputs.size(0))
        losses_d_fake.update(loss_d_fake.data, inputs.size(0))

        #lossRegressor = lossDepth + lossLap
        lossesRegressor.update(lossRegressor.data, batch_size)
        lossesDepth.update(lossDepth.mean().data, batch_size)
        lossesLap.update(lossLap.mean().data, batch_size)
        acces.update(acc[0], batch_size)

        if loader_idx % 50 == 0:
            npimg = im_to_numpy(inputs[0])
            io.imsave(os.path.join(args.checkpoint,"input.png"),npimg)
            show_joints3D(pts_img.detach()[0], outfn=os.path.join(args.checkpoint,"3dPoints.png"))
            show_joints3D(target.pts[0], outfn=os.path.join(args.checkpoint,"3dPoints_gt.png"))

            show_heatmap(target.heatmap_eyes.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"hm256_gt.png"))

            if train_fan:
                show_heatmap(out_hm.cpu().data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"hm256.png"))

                show_heatmap(target.heatmap64.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"hm64_gt.png"))
                show_heatmap(output[-1].cpu().data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"hm64.png"))
                sample_hm = sample_with_heatmap(inputs[0], output[-1][0].detach())
                io.imsave(os.path.join(args.checkpoint,"input-with-hm64.png"),sample_hm)
                sample_hm = sample_with_heatmap(inputs[0], target.heatmap64[0])
                io.imsave(os.path.join(args.checkpoint,"input-with-gt-hm64.png"),sample_hm)

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | lossFAN: {lossfan:.4f} | loss2D: {loss2d:.4f} | ' \
                     'Loss_d: {loss_d:.4f} | Loss_d_real: {loss_d_real:.4f} | Loss_d_fake: {loss_d_fake:.4f} | ' \
                     'LossRegressor: {lossReg:.4f} | lossDepth: {lossDepth:.4f} | lossLaplacian: {lossLap:.4f} | ' \
                     'Acc: {acc: .4f}'.format(
            batch=loader_idx + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            lossfan=lossesfan.avg,
            loss2d=losses2d.avg,
            loss_d=losses_d.avg,
            loss_d_real=losses_d_real.avg,
            loss_d_fake=losses_d_fake.avg,
            lossReg=lossesRegressor.avg,
            lossDepth=lossesDepth.avg,
            lossLap=lossesLap.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    return losses.avg, losses_d.avg, lossesRegressor.avg, lossesDepth.avg, lossesLap.avg, acces.avg

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
    for val_idx, (inputs, label, meta) in enumerate(loader):
        batch_size = inputs.size(0)
        target = Target._make(label)
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.to(device))
        target_var = target.heatmap64.to(device)
        target_var256 = target.heatmap_eyes.to(device)
        target_pts = target.pts.to(device)

        loss = torch.zeros([1], dtype=torch.float32)[0]
        if val_fan:
            out_hm, output = model.FAN(input_var)

            loss = 0
            for o in output:
                loss += criterion.hm(o, target_var)

            pts = heatmaps_to_coords(out_hm)

            pts = pts * 4

            loss += criterion.hm(pts, target_pts[:,:,:2])
        else:
            pts = target_pts[:,:,:2]
            output = target.heatmap64.unsqueeze(0)


        if val_fan and val_depth:
            heatmaps = make_gauss(pts, (256, 256), sigma=2)
            heatmaps = heatmaps.to(device)
        elif val_depth:
            heatmaps = target.heatmap_eyes.to(device)

        lossDepth = torch.zeros([1], dtype=torch.float32)[0]

        if val_depth:
            depth_inp = torch.cat((input_var, heatmaps), 1)
            depth_pred = model.Depth(depth_inp).detach()

            # intermediate supervision
            lossDepth = criterion.pts(depth_pred, target_pts[:,:,2])

            pts_img = torch.cat((pts.data, depth_pred.detach().data.unsqueeze(2)), 2).cpu()
        else:
            pts_img = torch.cat((pts.data, target_pts[:,:,2].unsqueeze(2)), 2).cpu()

        if val_idx % 50 == 0:
            npimg = im_to_numpy(inputs[0])
            io.imsave(os.path.join(args.checkpoint,"val-input.png"),npimg)
            show_joints3D(pts_img.detach()[0], outfn=os.path.join(args.checkpoint,"val_3dPoints.png"))
            show_joints3D(target.pts[0], outfn=os.path.join(args.checkpoint,"val_3dPoints_gt.png"))

            if val_fan:
                show_heatmap(output[-1].cpu().data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm64.png"))
                show_heatmap(target.heatmap64.data[0].unsqueeze(0), outname=os.path.join(args.checkpoint,"val_hm64_gt.png"))

                sample_hm = sample_with_heatmap(inputs[0], output[-1][0].detach())
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

        for n in range(batch_size):
            predictions[meta['index'][n], :, :] = pts_img[n, :, :]

        losses.update(loss.data, batch_size)
        losseslmk.update(lossDepth.data, batch_size)
        acces.update(acc[0], batch_size)

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

    auc = calc_metrics(all_dists, path=args.checkpoint, category='300W-Testset', method='Ours') # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))
    return losses.avg, losseslmk.avg, acces.avg, predictions, auc


if __name__ == '__main__':
    args = opts.argparser()

    main(args)
