import sys
is_py2 = sys.version[0] == '2'

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
if is_py2:
    import Queue as queue
else:
    import queue as queue
import queue
import threading

MAX_CAM=10
WINDOW_NAME="VRST 2019: DA-FAN"

# bufferless VideoCapture
class VideoCapture:
  def __init__(self, name):
    self.set_cam(name)
    self._stop = threading.Event()
    self.q = queue.Queue()
    self.t = threading.Thread(target=self._reader)
    self.t.daemon = True
    self.t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while (not self._stop.is_set()):
      ret, frame = self.cam.read()
      if not ret:
        self._stop.set()
        break
        
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)
    

  def read(self):
    if self._stop.is_set(): return None
    return self.q.get()

  def isOpened(self):
    return self.cam.isOpened()

  def set_cam(self, name):
    if hasattr(self, 'cap'):
        self.cam.release()
    self.cam = cv2.VideoCapture(name)

  def terminate(self):
    self._stop.set()
    self.t.join()



def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def testDevice(source):
   cap = cv2.VideoCapture(source)
   return cap is not None and cap.isOpened()

def getNextDevice(source):
    if not isInt(source): return source

    idx = (source + 1) % MAX_CAM
    while (idx != source):
        if testDevice(idx):
            return idx
        idx = (idx + 1) % MAX_CAM

    return source

def handle_keypress(state):
    k = cv2.waitKey(1) & 0xff
    if k == ord('q') or k == 27:
        state["run"] = False
        state["cam"].terminate()
        return

    if k == ord('c'):
        newCamID = getNextDevice(state["cam_id"])
        if newCamID != state["cam_id"]:
            state["cam_id"] = newCamID
            state["cam"].set_cam(state["cam_id"])

def annotate_frame(frame, preds, face_dets):
    [h, w] = frame.shape[:2]
    pil_image = Image.fromarray(frame)
    draw_image = ImageDraw.Draw(pil_image, 'RGBA')

    r = math.ceil(max(h, w) / 320)
    if preds is not None:
        for i in range(preds.shape[0]):
            face_det = face_dets[i]
            pred = preds[i]
            draw_image.rectangle(face_det, outline=(255, 0, 0))
            for j in range(pred.shape[0]):
                # d.point((pred[j,0],pred[j,1]), fill=255)
                x, y = pred[j, 0], pred[j, 1]
                draw_image.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0, 255), outline=(0, 0, 0))

    np_image = np.asarray(pil_image)

    return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

def setStateFromConfig(state, config):
    try:
        state["cam_id"] = int(config.camera_id)
    except:
        state["cam_id"] = config.camera_id

    state["max_faces"] = config.max_faces
    state["face_det"] = config.face_det

    return state

def read_image(state):
    image = state["cam"].read()
    
    if image is None:
        return None

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def defult_state():
    return {
        # run state
        "run":True,
        # Camera
        "cam_id":0,
        "cam":None,
        # tracking states
        "max_faces":0,
        "face_det":'sfd'
    }


def main(config):
    print(config)

    state = setStateFromConfig(defult_state(), config)

    # Run the 3D face alignment on a test image, with CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      network_size=NetworkSize.LARGE,
                                      device='cuda',
                                      max_faces=state["max_faces"],
                                      face_detector=state["face_det"] )


    # start up camera
    state["cam"] = VideoCapture(state["cam_id"])
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Program loop
    while state["run"]:
        image = read_image(state)

        if image is None:
            #state["run"] = False
            continue

        #start = time.time()
        preds, face_dets = fa.get_landmarks(image)
        #end = time.time()
        #print("Process Time: {}".format(end-start))

        annot_image = annotate_frame(image, preds, face_dets)

        cv2.imshow(WINDOW_NAME, annot_image)

        handle_keypress(state)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--camera-id', type=str, default=0, 
                      help='uint camera ID or file/path/to/vid.mp4. Default: 0')
    args.add_argument('--max-faces', type=int, default=0, 
                      help='Max faces detected, 0 is unlimited. Default: 0')
    args.add_argument('--face-det', type=str, choices=['sfd','dlib'], default='sfd', 
                      help='Face Detector to use SFD is better, dlib is faster. Default: sfd')

    config = args.parse_args()

    main(config)
