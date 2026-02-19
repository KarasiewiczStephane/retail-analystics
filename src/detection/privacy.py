"""Privacy pipeline for face detection and blurring.

Provides face detection using either YOLOv8-face or OpenCV Haar cascades,
and applies Gaussian blur to detected face regions. Ensures all exported
videos have faces anonymized.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    _YOLO = None

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """A detected face region.

    Attributes:
        bbox: Bounding box as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score.
    """

    bbox: tuple[int, int, int, int]
    confidence: float


class HaarFaceBlurrer:
    """Face blurrer using OpenCV Haar cascade detector.

    A lightweight alternative when YOLO face models are unavailable.

    Args:
        blur_intensity: Gaussian blur kernel size (forced to odd).
    """

    def __init__(self, blur_intensity: int = 51) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.blur_intensity = (
            blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        )

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces using Haar cascade.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of FaceDetection objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [
            FaceDetection(bbox=(int(x), int(y), int(x + w), int(y + h)), confidence=1.0)
            for (x, y, w, h) in faces
        ]

    def blur_face(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply Gaussian blur to a face region.

        Args:
            frame: BGR image (modified in place).
            bbox: Face region as ``(x1, y1, x2, y2)``.

        Returns:
            The frame with the face region blurred.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(
                roi, (self.blur_intensity, self.blur_intensity), 0
            )
            frame[y1:y2, x1:x2] = blurred
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect and blur all faces in a frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Copy of the frame with all faces blurred.
        """
        frame_copy = frame.copy()
        faces = self.detect_faces(frame_copy)
        for face in faces:
            frame_copy = self.blur_face(frame_copy, face.bbox)
        return frame_copy


class FaceBlurrer:
    """Face blurrer using a YOLO face detection model.

    Falls back to Haar cascade if the YOLO model cannot be loaded.

    Args:
        model_path: Path to YOLOv8-face model weights.
        confidence_threshold: Minimum confidence for face detection.
        blur_intensity: Gaussian blur kernel size (forced to odd).
    """

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.3,
        blur_intensity: int = 51,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.blur_intensity = (
            blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        )
        self._fallback = HaarFaceBlurrer(self.blur_intensity)
        self._use_yolo = False
        self.model = None

        if _YOLO is not None:
            try:
                self.model = _YOLO(model_path)
                self._use_yolo = True
                logger.info("FaceBlurrer using YOLO model: %s", model_path)
            except Exception:
                logger.warning(
                    "Failed to load YOLO face model '%s', using Haar cascade fallback",
                    model_path,
                )

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a frame.

        Uses YOLO if available, otherwise falls back to Haar cascades.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of FaceDetection objects.
        """
        if not self._use_yolo:
            return self._fallback.detect_faces(frame)

        from .detector import _to_numpy

        results = self.model(frame, verbose=False)[0]
        faces = []
        for box in results.boxes:
            conf = float(_to_numpy(box.conf).flat[0])
            if conf >= self.confidence_threshold:
                coords = _to_numpy(box.xyxy[0])
                x1, y1, x2, y2 = map(int, coords)
                faces.append(FaceDetection(bbox=(x1, y1, x2, y2), confidence=conf))
        return faces

    def blur_face(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply Gaussian blur to a face region.

        Args:
            frame: BGR image (modified in place).
            bbox: Face region as ``(x1, y1, x2, y2)``.

        Returns:
            The frame with the face region blurred.
        """
        return self._fallback.blur_face(frame, bbox)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect and blur all faces in a frame.

        Args:
            frame: BGR image.

        Returns:
            Copy of the frame with all faces blurred.
        """
        frame_copy = frame.copy()
        faces = self.detect_faces(frame_copy)
        for face in faces:
            frame_copy = self.blur_face(frame_copy, face.bbox)
        return frame_copy

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Process a video, blurring all detected faces.

        Args:
            input_path: Path to the input video.
            output_path: Path for the output video with faces blurred.
            progress_callback: Optional ``callback(current_frame, total_frames)``.

        Returns:
            Dictionary with ``frames_processed`` and ``faces_blurred`` counts.
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        faces_blurred = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            for face in faces:
                frame = self.blur_face(frame, face.bbox)
                faces_blurred += 1

            out.write(frame)
            frame_count += 1

            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        out.release()

        logger.info(
            "Privacy processing complete: %d frames, %d faces blurred",
            frame_count,
            faces_blurred,
        )
        return {"frames_processed": frame_count, "faces_blurred": faces_blurred}
