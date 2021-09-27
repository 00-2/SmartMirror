import cv2

class Computer_Man:
    
    def __init__(self, x0=0, y0=0, scale = 1.0, graph = {
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
        self.x0 = x0
        self.y0 = y0
        self.scale = scale
        self.graph = graph

    def draw_man(self, image, shape, colors=[
                                    (112,25,25),
                                    (255,0,0)
                                    ]):
        j=0
        for point in self.graph:
            # draw circles
            cv2.circle(image, shape[point], 1, colors[j%len(colors)], -1)
            for edges in self.graph[point]:
                # draw lines between circles 
                if(point<edges):
                    cv2.line(image, shape[point],shape[edges],colors[j%len(colors)])
                    j = j+1
