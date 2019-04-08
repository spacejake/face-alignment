import sys
import os

sys.path.append(os.path.abspath('..'))

import face_alignment
from face_alignment import NetworkSize

import socketserver
from struct import unpack, pack
from os import unlink
from messages.gen.python.messages import messages_pb2
import numpy as np
from scipy.misc import toimage
import threading
import google.protobuf.internal.decoder as decoder
import google.protobuf.internal.encoder as encoder
import time


class FANDetector:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, network_size=NetworkSize.SMALL, device='cuda:0', flip_input=True)
        

    def detect(self, image, detected_faces=None):
        #landmarks must be float32, as client expects

        start = time.time()
        pred = self.fa.get_landmarks(image, detected_faces)[-1].astype(np.float32)
        end = time.time()

        print("Time to process image {}".format(end-start))
        return pred


class DetectorSession(socketserver.BaseRequestHandler):

    def __init__(self, request, client_address, server, detector=None):
        self.detector = detector
        socketserver.BaseRequestHandler.__init__(self, request, client_address, server)
        return

    def detect(self, image, detected_faces=None):
        if self.detector is not None:
            return self.detector.detect(image, detected_faces)
        else:
            # send Fake data
            lmk = np.array([[1, 2, 0.3], [4, 5, 0.1],
                            [7, 8, 0.5], [10, 11, 0.3],
                            [13, 14, 0.6], [16, 17, 0.2],
                            [19, 20, 1], [21, 22, 0.2],
                            [23, 24, 0.1], [25, 26, 0.8]], np.float32)
            return lmk


    # Handle each session (connection with each client)
    def handle(self):
        print("Connected to Client: {}".format(self.client_address))

        while True:
            lmkreq = messages_pb2.LmkReq()
            try:
                lmkreq, size = self.readMessage(lmkreq)
                print("Message size recived: {}".format(size))
            except Exception as e:
                print("Error recv message: {}".format(e))
                #exit(1)
                break

            id = lmkreq.hdr.id
            imgShape = (lmkreq.hdr.width, lmkreq.hdr.height, lmkreq.hdr.channels)
            print("Message recived ID: {}".format(id))
            message_image = lmkreq.data.buffer
            img = np.frombuffer(message_image, dtype=np.uint8)
            img = np.reshape(img, imgShape)

            d = lmkreq.bbox
            detected_faces=None
            if d is not None:
                detected_faces = [[d.left, d.top, d.right, d.bottom]]

            # print("Face BBOX: {}".format(detected_faces))
            #toimage(img).show()

            landmarks = self.detect(img, detected_faces)

            lmkrsp = messages_pb2.LmkRsp()
            lmkrsp.hdr.id = id
            lmkrsp.dim.shape.extend(list(landmarks.shape))
            lmkrsp.data = np.ndarray.tobytes(landmarks)

            try:
                self.writeMessage(lmkrsp)
            except Exception as e:
                print("Error recv message: {}".format(e))
                #exit(1)
                break

        print("Closing client connection: {}".format(self.client_address))

    def writeDelimited(self, buffer):
        delimitedBuf = encoder._VarintBytes(len(buffer)) + buffer
        self.request.sendall(delimitedBuf)

    def writeMessage(self, message):
        # Output a message to be read with Java's parseDelimitedFrom
        import google.protobuf.internal.encoder as encoder
        serialized = message.SerializeToString()
        self.writeDelimited(serialized)

    def readDelimited(self):
        buf = []
        data = self.request.recv(8) #8 bytes is max Variant (for upto 64 bits)
        rCount = len(data)
        (size, position) = decoder._DecodeVarint(data, 0)

        buf.append(data)
        while rCount < size + position:
            data = self.request.recv(size + position - rCount)
            rCount += len(data)
            buf.append(data)

        return b''.join(buf), size, position

    def readMessage(self, message):
        data, size, position = self.readDelimited()
        message.ParseFromString(data[position:position + size])

        return message, size


class DecectorServer(socketserver.UnixStreamServer):
    def __init__(self, server_address, handler_class=DetectorSession):
        socketserver.UnixStreamServer.__init__(self, server_address, handler_class)
        self.detector = FANDetector()
        return

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass."""
        self.RequestHandlerClass(request, client_address, self, self.detector)


if __name__ == '__main__':
    address = 'socket'

    try:
        unlink(address)
    except OSError as e:
        pass

    print("Starting Server...")

    server = DecectorServer(address)
    serverThread = threading.Thread(target=server.serve_forever,
                                    name="LmkServer",
                                    daemon=threading.main_thread())
    serverThread.start()
    print("Listening...")

    input("Press <RETURN> to stop server\n")

    print("Shutting down Server...")
    server.shutdown()
    print("Exiting...")
    serverThread.join()



