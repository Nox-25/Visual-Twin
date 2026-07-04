import cv2
import numpy as np
import os
import glob
import time
from abc import ABC, abstractmethod

class FrameSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Returns a frame (numpy array, BGR) or None if error/EOF."""
        pass

    @abstractmethod
    def release(self):
        """Releases any acquired resources."""
        pass


class LiveCameraSource(FrameSource):
    def __init__(self, camera_index=0, timeout=2.0):
        self.cap = cv2.VideoCapture(camera_index)
        self.is_opened = False

        if not self.cap.isOpened():
            return

        start_time = time.time()
        # Try to read a frame within the timeout to ensure it works
        while (time.time() - start_time) < timeout:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.is_opened = True
                break
            time.sleep(0.1)

    def get_frame(self):
        if not self.is_opened:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()


class ImageFolderSource(FrameSource):
    def __init__(self, folder_path, loop=True, fps=1.0):
        self.folder_path = folder_path
        self.loop = loop
        self.fps = fps
        self.image_files = sorted(
            glob.glob(os.path.join(folder_path, '*.jpg')) +
            glob.glob(os.path.join(folder_path, '*.png'))
        )
        self.current_idx = 0
        self.last_frame_time = 0

    def get_frame(self):
        if not self.image_files:
            return None

        # Respect the fps setting
        current_time = time.time()
        if current_time - self.last_frame_time < (1.0 / self.fps):
            # Not enough time has passed, return the previous frame by not advancing
            pass

        if self.current_idx >= len(self.image_files):
            if self.loop:
                self.current_idx = 0
            else:
                return None

        img_path = self.image_files[self.current_idx]
        frame = cv2.imread(img_path)

        if current_time - self.last_frame_time >= (1.0 / self.fps):
            self.current_idx += 1
            self.last_frame_time = current_time

        return frame

    def release(self):
        pass


class SyntheticNoiseSource(FrameSource):
    def __init__(self, width=640, height=480, fps=1.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.last_frame_time = 0
        self.frame_count = 0

    def get_frame(self):
        current_time = time.time()
        if current_time - self.last_frame_time < (1.0 / self.fps):
            return self._last_frame if hasattr(self, '_last_frame') else None

        # Base texture: Gaussian noise over a base color
        base_color = np.array([120, 130, 125], dtype=np.uint8) # grayish background
        frame = np.full((self.height, self.width, 3), base_color, dtype=np.uint8)

        noise = np.random.normal(0, 10, (self.height, self.width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Inject artificial defects every few frames
        if self.frame_count % 3 == 1:
            # Add scratch (line stroke)
            x1, y1 = np.random.randint(0, self.width//2), np.random.randint(0, self.height)
            x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(-50, 50)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), thickness=2)
        elif self.frame_count % 3 == 2:
            # Add crack (random walk)
            x, y = np.random.randint(self.width//4, 3*self.width//4), np.random.randint(self.height//4, 3*self.height//4)
            for _ in range(10):
                nx = x + np.random.randint(-15, 15)
                ny = y + np.random.randint(5, 20)
                cv2.line(frame, (x, y), (nx, ny), (30, 30, 30), thickness=1)
                x, y = nx, ny

        self.frame_count += 1
        self.last_frame_time = current_time
        self._last_frame = frame
        return frame

    def release(self):
        pass


def get_camera(fallback_folder="data/images"):
    """Auto-fallback logic: LiveCamera -> ImageFolder -> SyntheticNoise"""
    print("Attempting to initialize LiveCameraSource...")
    cam = LiveCameraSource(timeout=2.0)
    if cam.is_opened:
        print("Live camera initialized successfully.")
        return cam

    print("Live camera failed. Falling back to ImageFolderSource...")
    if os.path.exists(fallback_folder):
        cam = ImageFolderSource(fallback_folder, fps=1.0)
        # Check if folder has images
        if cam.image_files:
            print(f"ImageFolderSource initialized with {len(cam.image_files)} images.")
            return cam

    print("Image folder empty or missing. Falling back to SyntheticNoiseSource...")
    cam = SyntheticNoiseSource(fps=1.0)
    return cam
