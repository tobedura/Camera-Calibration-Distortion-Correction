import cv2
import datetime
import os
import threading
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from filters import apply_canny


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    recording_started = pyqtSignal()

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self._recording = False
        self._writer = None
        self._lock = threading.Lock()
        self._width = 640
        self._height = 480
        self._fps = 30.0
        self._flip = False
        self._canny_record = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"카메라 {self.camera_index}를 열 수 없습니다.")
            return

        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._running = True
        first_frame_written = False
        consecutive_failures = 0
        max_failures = 10
        while self._running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.error.emit("프레임을 읽을 수 없습니다.")
                    break
                continue
            consecutive_failures = 0

            with self._lock:
                if self._flip:
                    frame = cv2.flip(frame, 1)

                if self._recording and self._writer is not None:
                    self._writer.write(apply_canny(frame) if self._canny_record else frame)
                    if not first_frame_written:
                        first_frame_written = True
                        self.recording_started.emit()
                elif not self._recording:
                    first_frame_written = False

            self.frame_ready.emit(frame.copy())

        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        cap.release()

    def stop(self):
        self._running = False
        self.wait()

    def start_recording(self, output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"recording_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            filename, fourcc, self._fps, (self._width, self._height)
        )
        if not writer.isOpened():
            self.error.emit(f"녹화 파일을 생성할 수 없습니다: {filename}")
            return filename
        with self._lock:
            self._writer = writer
            self._recording = True
        return filename

    def stop_recording(self):
        with self._lock:
            self._recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    @property
    def flip(self):
        with self._lock:
            return self._flip

    @flip.setter
    def flip(self, value: bool):
        with self._lock:
            self._flip = value

    @property
    def canny_record(self):
        with self._lock:
            return self._canny_record

    @canny_record.setter
    def canny_record(self, value: bool):
        with self._lock:
            self._canny_record = value

    @staticmethod
    def detect_cameras(max_index: int = 5) -> list:
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
