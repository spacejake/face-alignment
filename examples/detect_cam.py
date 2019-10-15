import sys
import os
import argparse
sys.path.append(os.path.abspath('..'))

import face_alignment
from face_alignment import NetworkSize
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import math
import time
from PIL import Image, ImageDraw
import cv2

def main(config):

    # Run the 3D face alignment on a test image, without CUDA.
    #fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      network_size=NetworkSize.LARGE,
                                      device='cuda',
                                      flip_input=False) #, face_detector='dlib')

    try:
        camID = int(config.camera_id)
    except:
        camID = config.camera_id

    cap = cv2.VideoCapture(camID)
    cv2.namedWindow("DA-FAN VRST 2019", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("face crop", cv2.WINDOW_NORMAL)

    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        #print("Frame shape: {}".format(image.shape))
        image = cv2.flip(image, 1)

        start = time.time()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        preds, face_dets = fa.get_landmarks(image)
        end = time.time()
        #print("Process Time: {}, Shape: {}".format(end-start, preds.shape))

        
        pil_image = Image.fromarray(image)
        frame = ImageDraw.Draw(pil_image, 'RGBA')

        if preds is not None:
            for i in range(preds.shape[0]):
                face_det = face_dets[i]
                pred = preds[i]
                frame.rectangle(face_det, outline=(255,0,0) )
                for j in range(pred.shape[0]):
                    # d.point((pred[j,0],pred[j,1]), fill=255)
                    x, y = pred[j,0], pred[j,1]
                    r = math.ceil(max(h, w)/320)
                    frame.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255), outline=(0,0,0))
        
        np_image = np.asarray(pil_image)

        cv2.imshow("DA-FAN VRST 2019", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
                break

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--camera-id', type=str, default=0)

    config = args.parse_args()

    main(config)
