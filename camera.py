"""Module 5a: threaded camera capture, so the main loop always gets the newest frame
instead of queuing behind MediaPipe's processing time."""
import threading
import cv2


class CameraThread:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # may be ignored by DSHOW; harmless
        self.lock = threading.Lock()
        self.frame, self.idx, self.running = None, 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """Grab frames continuously; keep only the latest one."""
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.idx = frame, self.idx + 1

    def read(self):
        """Return (frame_copy, frame_index); frame is None until the first grab."""
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.idx

    def release(self):
        self.running = False
        self.cap.release()
