import sys

import numpy as np
import cv2 as cv
import argparse

class App(object):

    def __init__(self, args):
        self.args = args

    def initializeTracker(self, image, trackerAlgorithm):
        while True:
            tracker = cv.TrackerMOSSE_create()
            # if trackerAlgorithm == 'mil':
            #     tracker = cv.TrackerMIL_create()
            # elif trackerAlgorithm == 'goturn':
            #     params = cv.TrackerGOTURN_Params()
            #     params.modelTxt = self.args.goturn
            #     params.modelBin = self.args.goturn_model
            #     tracker = cv.TrackerGOTURN_create(params)
            # elif trackerAlgorithm == 'dasiamrpn':
            #     params = cv.TrackerDaSiamRPN_Params()
            #     params.model = self.args.dasiamrpn_net
            #     params.kernel_cls1 = self.args.dasiamrpn_kernel_cls1
            #     params.kernel_r1 = self.args.dasiamrpn_kernel_r1
            #     tracker = cv.TrackerDaSiamRPN_create(params)
            # else:
            #     sys.exit("Tracker {} is not recognized. Please use one of three available: mil, goturn, dasiamrpn.".format(trackerAlgorithm))

            print('==> Select object ROI for tracker ...')
            bbox = cv.selectROI('tracking', image)
            print('ROI: {}'.format(bbox))

            try:
                tracker.init(image, bbox)
            except Exception as e:
                print('Unable to initialize tracker with requested bounding box. Is there any object?')
                print(e)
                print('Try again ...')
                continue

            return tracker

    def run(self):
        videoPath = self.args.input
        trackerAlgorithm = self.args.tracker_algo
        camera = cv.VideoCapture(0)
        if not camera.isOpened():
            sys.exit("Can't open video stream: {}".format(videoPath))

        ok, image = camera.read()
        if not ok:
            sys.exit("Can't read first frame")
        assert image is not None

        cv.namedWindow('tracking')
        tracker = self.initializeTracker(image, trackerAlgorithm)

        print("==> Tracking is started. Press 'SPACE' to re-initialize tracker or 'ESC' for exit...")

        while camera.isOpened():
            ok, image = camera.read()
            if not ok:
                print("Can't read frame")
                break

            ok, newbox = tracker.update(image)
            #print(ok, newbox)

            if ok:
                cv.rectangle(image, newbox, (200,0,0))

            cv.imshow("tracking", image)
            k = cv.waitKey(1)
            if k == 32:  # SPACE
                tracker = self.initializeTracker(image)
            if k == 27:  # ESC
                break

        print('Done')


if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser(description="Run tracker")
    parser.add_argument("--input", type=str, default="vtest.avi", help="Path to video source")
    parser.add_argument("--tracker_algo", type=str, default="dasiamrpn", help="One of three available tracking algorithms: mil, goturn, dasiamrpn")
    parser.add_argument("--goturn", type=str, default="goturn.prototxt", help="Path to GOTURN architecture")
    parser.add_argument("--goturn_model", type=str, default="goturn.caffemodel", help="Path to GOTERN model")
    parser.add_argument("--dasiamrpn_net", type=str, default="dasiamrpn_model.onnx", help="Path to onnx model of DaSiamRPN net")
    parser.add_argument("--dasiamrpn_kernel_r1", type=str, default="dasiamrpn_kernel_r1.onnx", help="Path to onnx model of DaSiamRPN kernel_r1")
    parser.add_argument("--dasiamrpn_kernel_cls1", type=str, default="dasiamrpn_kernel_cls1.onnx", help="Path to onnx model of DaSiamRPN kernel_cls1")
    parser.add_argument("--dasiamrpn_backend", type=int, default=0, help="Choose one of computation backends:\
                                                                           0: automatically (by default),\
                                                                           1: Halide language (http://halide-lang.org/),\
                                                                           2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit),\
                                                                           3: OpenCV implementation")
    parser.add_argument("--dasiamrpn_target", type=int, default=0, help="Choose one of target computation devices:\
                                                                         0: CPU target (by default),\
                                                                         1: OpenCL,\
                                                                         2: OpenCL fp16 (half-float precision),\
                                                                         3: VPU")
    args = parser.parse_args()
    App(args).run()
    cv.destroyAllWindows()