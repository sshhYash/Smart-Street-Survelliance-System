"""Video input: files are read frame by frame; live streams (RTSP / webcam)
use a reader thread that keeps only the latest frame and reconnects on drops."""
import threading
import time
from pathlib import Path

import cv2


class VideoSource:
    def __init__(self, source):
        source = str(source)
        self.source = int(source) if source.isdigit() else source
        self.is_file = isinstance(self.source, str) and Path(self.source).exists()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 120 else 25.0
        self.online = True
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        if not self.is_file:
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self.online = False
                self.cap.release()
                time.sleep(3)
                self.cap = cv2.VideoCapture(self.source)
                continue
            self.online = True
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the next frame, or None when a video file ends."""
        if self.is_file:
            ok, frame = self.cap.read()
            return frame if ok else None
        while not self._stop:
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                return frame
            time.sleep(0.005)
        return None

    def release(self):
        self._stop = True
        time.sleep(0.05)
        self.cap.release()
