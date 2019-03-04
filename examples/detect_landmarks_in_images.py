import face_alignment
from face_alignment import NetworkSize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import os
import time

# Run the 3D face alignment on a test image, without CUDA.
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

#directory = '../test/assets/LandmarkTests'
#directory = '../test/assets/smallTest'
directory = '../test/assets/kinectTest'

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        print('Reading image {}'.format(os.path.join(directory, filename)))
        input = io.imread(os.path.join(directory, filename))
        start = time.time()
        preds = fa.get_landmarks(input)[-1]
        end = time.time()
        
        #TODO: Make this nice
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input)
        ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
        ax.axis('off')
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
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
        #plt.show()

        plt.savefig('output-{}'.format(filename))
        print("File {}, process Time: {}".format(filename, end-start))
        plt.close()
        break
    else:
        print('ignoring file {}'.format(os.path.join(directory, filename)))
        #continue
        break


#plt.savefig('output.png')
#plt.savefig('output-me.png')
#plt.savefig('output-blink.png')