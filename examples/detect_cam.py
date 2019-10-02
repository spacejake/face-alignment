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
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                      network_size=NetworkSize.LARGE,
                                      device='cuda',
                                      flip_input=False)

    if len(sys.argv) != 2:
        print("usage:%s (cameraID | filename) Detect faces\
    in the video example:%s 0" % (sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        camID = int(config.camera_id)
    except:
        camID = config.camera_id

    cap = cv2.VideoCapture(camID)
    cv2.namedWindow("DA-FAN VRST 2019", cv2.WINDOW_NORMAL)

    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        # print(h, w)
        # input = cv2.flip(image, 1)

        start = time.time()
        preds = fa.get_landmarks(input)
        end = time.time()
        print("Process Time: {}".format(end-start))

        pil_image = Image.fromarray(input)
        frame = ImageDraw.Draw(pil_image, 'RGBA')

        if preds is not None:
            for pred in preds:

                for i in range(preds.shape[0]):
                    # d.point((preds[i,0],preds[i,1]), fill=255)
                    x, y = preds[i,0], preds[i,1]
                    r = math.ceil(max(h, w)/320)
                    frame.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255, 255), outline=(0,0,0))

        cv2.imshow("DA-FAN VRST 2019", np.array(frame))
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
                break

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--camera-id', type=int)

    config = args.parse_args()

    main(config)
