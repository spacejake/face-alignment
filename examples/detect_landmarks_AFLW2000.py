import sys
import os
sys.path.append(os.path.abspath('..'))

import face_alignment
from face_alignment import NetworkSize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import os
import time

import torch
from face_alignment.datasets.common import Target
from face_alignment.datasets.AFLW2000 import AFLW2000
import face_alignment.util.opts as opts
from face_alignment.util.imutils import im_to_numpy

# Run the 3D face alignment on a test image, without CUDA.
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, network_size=NetworkSize.SMALL,
                                  device='cuda', flip_input=False)

#directory = '../test/assets/LandmarkTests'
#directory = '../test/assets/smallTest'
directory = '../test/assets/kinectTest'

args = opts.argparser()
datasetLoader = AFLW2000
crop_win = None
loader = torch.utils.data.DataLoader(
    datasetLoader(args, 'test', demo=True),
    batch_size=1,
    # shuffle=True,
    num_workers=1,
    pin_memory=True)
for i, data in enumerate(loader):
    input, label, meta = data
    target = Target._make(label)
    img_fn = meta['img_fn'][0]
    orig_img = im_to_numpy(meta['orig_img'][0])
    orig_pts = meta['orig_pts'][0].numpy()

    print('Reading image {}'.format(os.path.join(img_fn)))
    start = time.time()
    preds = fa.get_landmarks_from_face_image(input, target.center, target.scale).numpy()
    end = time.time()

    #TODO: Make this nice
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(orig_img)
    # surf = ax.scatter(orig_pts[:,0],orig_pts[:,1],c="r", marker='o',s=1)
    surf = ax.scatter(preds[:,0],preds[:,1],c="w", marker='o',s=8, edgecolor='k')
    # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.axis('off')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(orig_pts[:,0]*1.2,orig_pts[:,1],orig_pts[:,2],c="red", alpha=0.3, edgecolor='r')
    ax.plot3D(orig_pts[:17,0]*1.2,orig_pts[:17,1], orig_pts[:17,2], color='red', alpha=0.3 )
    ax.plot3D(orig_pts[17:22,0]*1.2,orig_pts[17:22,1],orig_pts[17:22,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[22:27,0]*1.2,orig_pts[22:27,1],orig_pts[22:27,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[27:31,0]*1.2,orig_pts[27:31,1],orig_pts[27:31,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[31:36,0]*1.2,orig_pts[31:36,1],orig_pts[31:36,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[36:42,0]*1.2,orig_pts[36:42,1],orig_pts[36:42,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[42:48,0]*1.2,orig_pts[42:48,1],orig_pts[42:48,2], color='red', alpha=0.3)
    ax.plot3D(orig_pts[48:,0]*1.2,orig_pts[48:,1],orig_pts[48:,2], color='red', alpha=0.3)

    surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
    ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
    ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
    ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
    ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
    ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
    ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
    ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    # plt.show()

    plt.savefig('output-{}'.format(img_fn[-14:]))
    print("File {}, process Time: {}".format(img_fn, end-start))
    plt.close()
    # break

#plt.savefig('output.png')
#plt.savefig('output-me.png')
#plt.savefig('output-blink.png')
