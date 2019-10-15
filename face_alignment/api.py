from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
from skimage import io
from skimage import color
import numpy as np
import cv2
import time
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *
from .util.gdrivedl import download, mkdir_safe
from .util.heatmap import make_gauss, heatmaps_to_coords

class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    TINY = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

models_urls = {
    '3DFAN-4': '1c4JLRAUFWWdzLGM6EW00VigWICZYIww6',
    '3DFAN-2': '16WZC28wWI7viWlmMILcGKzcyHBoBv2CI',
    'depth': '1jK8zsNsTRKtkeM3OhEvFZZNlWiKywAcL',
    #########################
    'FPFAN-4': '',
    'FPFAN-2': '',
    'FP-depth': '',
    #########################
    'DAFAN-4': '1NrLf4MtplPHCIoFp4qOVUjJ8dptbWipQ',
    'DAFAN-2': '1fbcCVw9XGeZgHWs4icg84in4Q82LcK7c',
    'DA-depth': '12JXxeUeiwvGxr3wPiPIIJLB_CngGImV7',
    'DAFAN-HMD-4': '',
    'DAFAN-HMD-d1.5-4': '',
    'DAFAN-MC-d1.5-4': '',
}

#models_chkpts = {
#    '3DFAN-2': '../ckpnt-fan256-small/checkpointFAN.pth.tar',
#    #'depth': '../ckpnt-laplacian-v1.2/checkpointDepth.pth.tar',
#    'depth': '../ckpnt-3DFAN-align/checkpointDepth.pth.tar',
#}

ckpnt_dir = "checkpoints"

def load_checkpoint(network_name):
    print("Loding Network Model {}...".format(network_name))

    ckpnt_fn = "{}.pth.tar".format(network_name)
    ckpnt_filepath = os.path.join(ckpnt_dir, ckpnt_fn)

    if not os.path.exists(ckpnt_filepath):
        print("Downloading...")
        mkdir_safe(ckpnt_dir)

        download(models_urls[network_name], ckpnt_filepath)

    checkpoint = torch.load(ckpnt_filepath)
    return checkpoint

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        print("My Version is Running!!")

        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        print("Get Face Detector...")
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)

        if landmarks_type == LandmarksType._2D:
            network_name = 'DAFAN-' + str(network_size)
        else:
            network_name = 'DAFAN-' + str(network_size)
        
        fan_checkpoint = load_checkpoint(network_name)
        #fan_weights = fan_checkpoint['state_dict']
        fan_weights = {
                k.replace('module.', ''): v for k,
                v in fan_checkpoint['state_dict'].items()}

        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_weights = load_checkpoint('DA-depth')
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if detected_faces is None:
            #start = time.time()
            detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())
            #print("Face Detection: {}s".format(time.time()-start))

        if len(detected_faces) == 0:
            #print("Warning: No faces were detected.")
            return None, None

        torch.set_grad_enabled(False)

        inp_b = None
        center_b = None
        scale_b = None
        landmarks = None

        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = torch.FloatTensor([(d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale])

            inp = crop(image, center, scale)
            #cv2.imshow("face crop", cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))

            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()
            inp.div_(255.0)

            if inp_b is None:
                inp_b = inp.unsqueeze_(0)
                center_b = center.unsqueeze(0)
                scale_b = scale.unsqueeze(0)
            else:
                inp_b = torch.cat((inp_b, inp.unsqueeze_(0)),0)
                center_b = torch.cat((center_b, center.unsqueeze(0)),0)
                scale_b = torch.cat((scale_b, scale.unsqueeze(0)),0)

        if inp_b is not None:
            #start = time.time()
            pts_img = self.get_landmarks_from_face_image(inp_b, center_b, scale_b)
            #print("Landmark Detection: {}s".format(time.time()-start))
            landmarks = pts_img.numpy()

        return landmarks, detected_faces


    def get_landmarks_from_face_image(self, input, center, scale):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """

        input = input.to(self.device)

        out, _ = self.face_alignment_net(input)
        out = out.detach()
        if self.flip_input:
            flip_out_hm, _= self.face_alignment_net(flip(input))
            out += flip(flip_out_hm.detach(), is_label=True)

        #out = out.cpu()

        pts = heatmaps_to_coords(out)
        pts_img = scale_preds(pts, center, scale)
        pts, pts_img = pts * 4, pts_img

        if self.landmarks_type == LandmarksType._3D:
            heatmaps = make_gauss(pts, (256, 256), sigma=2).unsqueeze(0)
            heatmaps = heatmaps.to(self.device)

            depth_pred = self.depth_prediciton_net(
                torch.cat((input, heatmaps), 1)).data.cpu().view(68, 1)
            pts_img = torch.cat(
                (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

        landmarks = pts_img

        return landmarks

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True):
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            preds = self.get_landmarks_from_image(image, bounding_boxes)
            predictions[image_path] = preds

        return predictions

    @staticmethod
    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)
