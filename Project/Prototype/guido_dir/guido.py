from computer_man import Computer_Man
import cv2
import dlib
from pathlib import Path
from imutils import face_utils




path = '../predictors/shape_predictor_68_face_landmarks.dat' #path to shape_predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

# data from web cam
cap = cv2.VideoCapture(0)
while True:
    # black background
    bg = cv2.imread("../add_data_dir/background_black.jpg")
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
	    # create computer man 
        man = Computer_Man()
        man.draw_man(bg,shape)

    # Show the image
    cv2.imshow("Output", bg)
    


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()