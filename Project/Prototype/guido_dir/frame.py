import screeninfo
class Frame:
    def __init__(self, monitor = screeninfo.get_monitors()[0]):
        self.monitor_height = monitor.height
        self.monitor_width = monitor.width
        