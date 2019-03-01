import numpy as np
import scipy.io as sio
import skimage.io as skio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchfile
from sklearn.metrics import mean_squared_error

# modelMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/HELEN/HELEN_10405146_1_0.mat')
#lmkMat = torchfile.load('/home/jake/face/face-tracking/dataset/landmarks/AFW/AFW_134212_1_5_pts.t7')

# IBUG
input = skio.imread("/home/jake/face/face-tracking/dataset/300W_LP/IBUG/IBUG_image_003_1_0.jpg")
lmk2dMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks/IBUG/IBUG_image_003_1_0_pts.mat')
lmkMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks3d/IBUG/IBUG_image_003_1_0_pts.mat')

# AFW
# input = skio.imread("/home/jake/face/face-tracking/dataset/300W_LP/AFW/AFW_134212_1_5.jpg")
# lmk2dMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks/AFW/AFW_134212_1_5_pts.mat')
# lmkMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks3d/AFW/AFW_134212_1_5_pts.mat')

# HELEN
# input = skio.imread("/home/jake/face/face-tracking/dataset/300W_LP/HELEN/HELEN_10405146_1_0.jpg")
# lmk2dMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks/HELEN/HELEN_10405146_1_0_pts.mat')
# lmkMat = sio.loadmat('/home/jake/face/face-tracking/dataset/300W_LP/landmarks3d/HELEN/HELEN_10405146_1_0_pts.mat')

#lmk_2d = lmkMat['pts_2d']
lmk_2d = lmk2dMat['pts_3d']
lmk_3d = lmkMat['pts_3d']
#lmk_3d = lmkMat[0]
lmk = lmk_3d

#mse = mean_squared_error(lmk_3d[:, 0:2], lmk_2d)
lmkError = (np.square(lmk_3d[:, 0:2] - lmk_2d)).mean(axis=1)
#print(lmk)
print("Error between 2D fitted and annotated lmks: {}".format(lmkError))

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input)
ax.plot(lmk[0:17, 0], lmk[0:17, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[17:22, 0], lmk[17:22, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[22:27, 0], lmk[22:27, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[27:31, 0], lmk[27:31, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[31:36, 0], lmk[31:36, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[36:42, 0], lmk[36:42, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[42:48, 0], lmk[42:48, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[48:60, 0], lmk[48:60, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.plot(lmk[60:68, 0], lmk[60:68, 1], marker='o', markersize=1, linestyle='-', color='w', lw=1)
ax.axis('off')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.scatter(lmk[:, 0] * 1.2, lmk[:, 1], lmk[:, 2], c="cyan", alpha=1.0, edgecolor='b')
ax.plot3D(lmk[:17, 0] * 1.2, lmk[:17, 1], lmk[:17, 2], color='blue')
ax.plot3D(lmk[17:22, 0] * 1.2, lmk[17:22, 1], lmk[17:22, 2], color='blue')
ax.plot3D(lmk[22:27, 0] * 1.2, lmk[22:27, 1], lmk[22:27, 2], color='blue')
ax.plot3D(lmk[27:31, 0] * 1.2, lmk[27:31, 1], lmk[27:31, 2], color='blue')
ax.plot3D(lmk[31:36, 0] * 1.2, lmk[31:36, 1], lmk[31:36, 2], color='blue')
ax.plot3D(lmk[36:42, 0] * 1.2, lmk[36:42, 1], lmk[36:42, 2], color='blue')
ax.plot3D(lmk[42:48, 0] * 1.2, lmk[42:48, 1], lmk[42:48, 2], color='blue')
ax.plot3D(lmk[48:, 0] * 1.2, lmk[48:, 1], lmk[48:, 2], color='blue')

ax.view_init(elev=90., azim=90., )
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)

#plt.savefig('output-{}'.format(filename))
#print("File {}, process Time: {}".format(filename, end - start))
plt.close()