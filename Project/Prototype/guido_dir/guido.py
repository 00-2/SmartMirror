from computer_man import Computer_Man
import cv2
import dlib
from pathlib import Path
from imutils import face_utils
import screeninfo
import time
from paz.pipelines import MiniXceptionFER



path = '../predictors/shape_predictor_68_face_landmarks.dat' #path to shape_predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

monitor = screeninfo.get_monitors()[0]
monitor_height, monitor_width = monitor.height,monitor.width
x_parallax = monitor_width/2
y_parallax = monitor_height/2
scale = 2

# data from web cam
cap = cv2.VideoCapture(0)
# we work with one person
arr_of_mood = []
max_length_arr_of_mood = 200
# message of staying in vision
message = 'please stay in my field of vision'
message_length = len(message)


classify = MiniXceptionFER()

while len(arr_of_mood)<max_length_arr_of_mood:
    time.sleep(0.02)
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
        man = Computer_Man(x0=x_parallax,y0 = y_parallax, scale=scale)
        man.draw_man(bg,shape)
    cv2.putText(bg,message,(monitor_width - 700, 100),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
    # success percentage
     
    current_status =  int(len(arr_of_mood)/max_length_arr_of_mood*10)
    cv2.putText(bg,(">" + "="*current_status) + ">",(monitor_width - 700, 150),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0))   
    # add mood into array
    inference = classify(image)
    arr_of_mood.append(inference['class_name'])
    # Show the image
    cv2.imshow("Output", bg)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

# we found out mood

# now may drop that to server../find mood and make the animation
#thread 1 find mood
#thread 2 
color = 0
while(color<255):
    bg = cv2.imread("../add_data_dir/background_black.jpg")
    cv2.putText(bg,"Some magic..",(200, 200),cv2.FONT_HERSHEY_SIMPLEX, 1,(color,color,color))   
    color = color+1
    cv2.imshow("Output",bg)
    time.sleep(0.001)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()