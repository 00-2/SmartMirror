import cv2
import numpy
from frame import Frame
class Computer_Man(Frame):
    def __init__(self,x0=0,y0=0, scale = 1,graph = {
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
    }):
        Frame.__init__(self)
        self.x0 = x0
        self.y0 = -y0#cause in right + , down +, we starts from down
        self.scale = scale
        self.graph = graph

    def draw_man(self, image, shape, colors=[
                                    (112,25,25),
                                    (255,0,0)
                                    ]):
        j=0
        # move in left down corner
        # will mv to (0,0) by moving x from 1 and y from 9
        x_to_zero,y_tmp = shape[0]
        x_tmp, y_to_zero = shape[8]
        parallax_to_zero = numpy.array([x_to_zero,y_to_zero-self.monitor_height],dtype=numpy.int32)
        
        # parallax from x0, y0
        parallax = numpy.array([self.x0,self.y0],dtype=numpy.int32)
        for point in self.graph:            
            # draw circles
            result_point_coordinates = (shape[point]-parallax_to_zero)+parallax 
            cv2.circle(image, result_point_coordinates, 1, colors[j%len(colors)], -1)
            for edges in self.graph[point]:
                # draw lines between circles 
                if(point<edges):
                    result_edges_coordinates = (shape[edges]-parallax_to_zero)+parallax
                    cv2.line(image, result_point_coordinates, result_edges_coordinates,colors[j%len(colors)])
                    j = j+1
    def to_middle(self):
        self.x0 = self.monitor_width/2
        self.y0 = -self.monitor_height/2