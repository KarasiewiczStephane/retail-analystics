"""YOLOv8 person detection pipeline.

Uses a pre-trained YOLOv8 model on COCO to detect persons (class_id=0)
in video frames with configurable confidence thresholds.
"""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _to_numpy(tensor_or_array: "np.ndarray") -> np.ndarray:
    """Convert a YOLO tensor or numpy array to a plain numpy array.

    Args:
        tensor_or_array: Either a PyTorch tensor or a numpy array.

    Returns:
        Numpy array with the same data.
    """
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


@dataclass
class Detection:
    """A single person detection in a video frame.

    Attributes:
        bbox: Bounding box coordinates as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score between 0 and 1.
        class_id: COCO class identifier (0 for person).
        centroid: Center point of the bounding box as ``(cx, cy)``.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    centroid: tuple[float, float]


class PersonDetector:
    """YOLOv8-based person detector.

    Filters detections to only return the person class (class_id=0)
    above a configurable confidence threshold.

    Args:
        model_path: Path to the YOLOv8 model weights.
        confidence_threshold: Minimum confidence for a detection to be kept.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        logger.info(
            "PersonDetector initialized: model=%s, threshold=%.2f",
            model_path,
            confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run person detection on a single frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of Detection objects for persons above the confidence threshold.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(_to_numpy(box.cls).flat[0])
            conf = float(_to_numpy(box.conf).flat[0])
            if cls_id == self.PERSON_CLASS_ID and conf >= self.confidence_threshold:
                coords = _to_numpy(box.xyxy[0])
                x1, y1, x2, y2 = coords
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                        class_id=self.PERSON_CLASS_ID,
                        centroid=(float(cx), float(cy)),
                    )
                )

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run person detection on a batch of frames.

        Args:
            frames: List of BGR images as numpy arrays.

        Returns:
            List of detection lists, one per input frame.
        """
        all_results = self.model(frames, verbose=False)
        batch_detections = []

        for results in all_results:
            frame_detections = []
            for box in results.boxes:
                cls_id = int(_to_numpy(box.cls).flat[0])
                conf = float(_to_numpy(box.conf).flat[0])
                if cls_id == self.PERSON_CLASS_ID and conf >= self.confidence_threshold:
                    coords = _to_numpy(box.xyxy[0])
                    x1, y1, x2, y2 = coords
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    frame_detections.append(
                        Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf,
                            class_id=self.PERSON_CLASS_ID,
                            centroid=(float(cx), float(cy)),
                        )
                    )
            batch_detections.append(frame_detections)

        return batch_detections
