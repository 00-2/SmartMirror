from imutils import face_utils
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    mar = (1.0)/(4.0*A)
    return mar

cap = cv2.VideoCapture(0)
blink_count = 0
is_smiling = 0
blink_count_speed = 1
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    bg = cv2.imread("background.png")
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR+right_EAR)/2
        cv2.putText(bg, "Blinks: {}".format(blink_count), (470, 290),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)        
        
        ellipse_len=25
        if avg_EAR < 0.22:
            ellipse_len=7
            blink_count +=1
        
        # Draw on our image, all the finded cordinate points (x,y) 
        cv2.circle(bg, (shape[33][0], shape[33][1]-15), shape[33][0]-shape[4][0]+30, (0, 255, 255), -1)
        cv2.ellipse(bg, (shape[46][0]-7, shape[46][1]), (7, ellipse_len), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(bg, (shape[40][0]-3, shape[40][1]), (7, ellipse_len), 0, 0, 360, (0, 0, 0), -1)
        
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        mShape = shape[mStart:mEnd]
        mouth = cv2.convexHull(mShape)
        cv2.drawContours(bg, [mouth], -1, (0, 0, 0), 3)
        
        mar = mouth_aspect_ratio(shape[48:60])
        
        if mar<0.0040:
            is_smiling = 1
            blink_count+=blink_count_speed
            cv2.putText(bg, "Keep Smiling!", (470, 360+(i*40)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            is_smiling = 0
            cv2.putText(bg, "Please Smile", (470, 360+(i*40)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if is_smiling==1:
            blink_count_speed+=0.01
        else:
            blink_count_speed = 1
    if blink_count_speed>=1.5:
            cv2.putText(bg, "Wow!Bonus for Smiling!:{}x".format(blink_count_speed), (470, 400+(i*40)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    
    
    # Show the image
    cv2.imshow("Output", bg)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()