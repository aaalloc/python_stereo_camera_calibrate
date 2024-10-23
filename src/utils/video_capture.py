import cv2
import threading
import queue


class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.state = True
        self.running = True  # Flag to indicate if the thread should keep running
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def set_cap_flag(self, propId, value):
        self.cap.set(propId, value)

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.state = ret

    def read(self):
        return self.state, self.q.get()

    def stop(self):
        self.running = False
        self.t.join()  # Wait for the thread to exit
