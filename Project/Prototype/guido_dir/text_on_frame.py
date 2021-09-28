import cv2
from frame import Frame
class Text_On_Frame(Frame):
    def __init__(self, org=(0, Frame.monitor_heigth), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255)):
        pass
    def put_text_in_middle(self, frame, message):
        cv2.putText(frame,message, self.org, self.fontFace, self.fontScale, self.color)
