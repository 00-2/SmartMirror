import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER
from paz.pipelines import MiniXceptionFER
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    classify = MiniXceptionFER()
    
    cap = cv2.VideoCapture(0)
    while True:
        bg = cv2.imread("background_black.jpg")
        # Getting out image by webcam 
        _, image = cap.read()

    #    apply directly to an image (numpy-array)
        inference = classify(image)
        print(inference)
    