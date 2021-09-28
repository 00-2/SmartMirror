import cv2
from frame import Frame
class Text_On_Frame(Frame):
    shift = 50
    count_of_texts = 0
    x0 = 0
    y0 = 0
    def __init__(self, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255)):
        Frame.__init__(self)
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.x0 = self.monitor_width-700
        self.y0 = 150
    def put(self, frame, message):
        org = (self.x0,self.y0+self.count_of_texts*self.shift)
        self.shift = self.shift+1
        cv2.putText(frame,message, org, self.fontFace, self.fontScale, self.color)
