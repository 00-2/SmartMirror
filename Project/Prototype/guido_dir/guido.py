from text_on_frame import Text_On_Frame
from computer_man import Computer_Man
import cv2
import dlib
from pathlib import Path
from imutils import face_utils
import time
from paz.pipelines import MiniXceptionFER
from frame import Frame


path = '../predictors/shape_predictor_68_face_landmarks.dat' #path to shape_predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)


# data from web cam
cap = cv2.VideoCapture(0)
# we work with one person
arr_of_mood = []
max_length_arr_of_mood = 20
# message of staying in vision
message = 'please stay in my field of vision'
message_length = len(message)

frame = Frame()
man = Computer_Man()
percentage_of_success = Text_On_Frame()

classify = MiniXceptionFER()

# i - for get mood each 10 frames
i = 0
while len(arr_of_mood)<max_length_arr_of_mood:
    time.sleep(0.002)
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
        man.draw_man(bg,shape)

    # success percentage 
    percentage_of_success.put(bg,message)
    current_status =  (1)
    percentage_of_success.put(bg,("*"*current_status) + ">")
    # add mood into array
    if i % 10 == 0:
        inference = classify(image)
        arr_of_mood.append(inference['class_name'])
    # next Frame
    i = (i+1)%10
    # Show the image
    cv2.imshow("Output", bg)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

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