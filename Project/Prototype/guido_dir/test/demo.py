from threading import Event, Thread, Timer
import cv2, time

import dlib
from imutils import face_utils
import sys

from paz.pipelines import MiniXceptionFER

sys.path.append('/home/x00/repos/SmartMirror/Project/Prototype/guido_dir')
from text_on_frame import Text_On_Frame
from computer_man import Computer_Man

message = 'please stay in my field of vision'
percentage_of_success = Text_On_Frame()

percentage_of_success.length_of_message = percentage_of_success.length_of_text(message)[0][0]

current_length_percentage_bar = 0
max_length_arr_of_mood = 200
max_length_percentage_bar = percentage_of_success.get_length_of_percentage_bar()
count_of_moods_for_one_symbol = max_length_arr_of_mood/max_length_percentage_bar

class VideoStreamWidget(object):
    
    

    path = '../predictors/shape_predictor_68_face_landmarks.dat' #path to shape_predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)
    arr_of_moods = []

    man = Computer_Man()
    classify = MiniXceptionFER()
    stop_flag = Event
    
    
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread_get_frame = Thread(target=self.update, args=())
        self.thread_get_frame.daemon = True
        self.thread_get_frame.start()
        self.thread_get_mood = Thread(target = self.get_mood, args=())
        self.thread_get_mood.start()
    
    def get_mood(self):
        while len(self.arr_of_moods)<max_length_arr_of_mood:
            try:
                frame = self.frame
                inference = self.classify(frame)
                self.arr_of_moods.append(inference['class_name'])
            except AttributeError:
                pass
            time.sleep(0.02)

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        bg = cv2.imread("../add_data_dir/background_black.jpg")
        # Display frames in main program
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Get faces into webcam's image
        rects = self.detector(gray, 0)
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
	        # create computer man 
            self.man.to_middle()
            self.man.draw_man(bg,shape)
        percentage_of_success.put(bg,message)
        percentage_of_success.put(bg,("*"*(int)(len(self.arr_of_moods)/count_of_moods_for_one_symbol) + ">"))
        percentage_of_success.count_of_texts = 0
        cv2.imshow('frame', bg)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass