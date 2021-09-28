import cv2
from frame import Frame
class Text_On_Frame(Frame):
    def __init__(self, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255)):
        Frame.__init__(self)
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.x0 = self.monitor_width
        self.y0 = 150
        self.shift = 50
        self.count_of_texts = 0
        self.length_of_message = 0
    def length_of_text(self,message):
        return (cv2.getTextSize(message,self.fontFace,self.fontScale,1))
    
    def set_length_of_message(self, length):
        self.length_of_message = length
    
    def put(self, frame, message):
        #get length of message
        if (self.length_of_message==0):
            self.length_of_message= self.length_of_text(message)[0][0]
        #mv x0 for length + shift <-
        #mv y0 for count_of_texts*50 \/
        org = ( self.x0-self.length_of_message-self.shift,
                self.y0+self.count_of_texts*self.shift
        )
        self.count_of_texts = self.count_of_texts+1
        #put text
        cv2.putText(frame,message, org, self.fontFace, self.fontScale, self.color)


    def get_length_of_percentage_bar(self, head = ">", tail = "*", mid = "*"):
        str = head
        while self.length_of_text(str)[0][0]<self.length_of_message:
            str=mid+str
        return(len(str))
