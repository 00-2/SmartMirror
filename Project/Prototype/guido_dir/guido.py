#we have a head, we highlight pos, where text can be putted

import cv2
import dlib
from pathlib import Path
from imutils import face_utils
import numpy


path = Path().absolute().parent #path to Prototype
if (str(path.name) == 'Project'):
    path = path.joinpath('Prototype')

path = str(path.joinpath('predictors/shape_predictor_68_face_landmarks.dat')) #path to shape_predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

graph = { 
            0 : [17,3,36,48,31],
            3 : [0, 7,48,58],
            7 : [3,9,57,58],
            9 : [7,13,56,57],
            13 : [9,16,54,56],
            16 : [13,26,35,45,54],
            17 : [0,36],
            26 : [16, 45],
            27 : [30],
            30 : [31,35,27],
            31 : [0,30,40,48],
            35 : [16, 30, 42 , 54],
            36 : [0, 17,37,41],
            37 : [36],
            38: [39],
            39: [38,40],
            40: [39, 31],
            41: [36],
            42: [35,43,47],
            43: [42],
            44: [45],
            45: [46],
            47: [42],
            48: [0,3,31,50,58,61,67],
            50: [48,61],
            51: [61, 63],
            52: [54,63],
            54: [13,16,35,52,56,63,65],
            56: [9,13,54,65],
            57: [7,9,57],
            58: [3,7,48,67],
            61: [48,50,51],
            63: [51,52,64],
            65: [54,56,57],
            67: [48,57,58]
        }
        
colors = [(112,25,25), (255,0,0)]

cap = cv2.VideoCapture(0)
while True:
    bg = cv2.imread("background_black.jpg")
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
	    # loop over the (x, y)-coordinates for the facial landmarks
	    # and draw them on the image
        #for (x, y) in shape:
            #cv2.circle(bg, (x, y), 2, (238, 95, 26), -1)
        # making lines between some circles
        # lines from 1
        
        j=0
        for point in graph:
            for edges in graph[point]:
                if(point<edges):
                    cv2.line(bg, shape[point],shape[edges],colors[j%len(colors)])
                    j = j+1

        
        


    # Show the image
    cv2.imshow("Output", bg)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()