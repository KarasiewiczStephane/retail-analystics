"""Tests for the YOLOv8 person detection pipeline."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from src.detection.detector import Detection, PersonDetector


@dataclass
class FakeBox:
    """Fake bounding box for mocking YOLO results."""

    xyxy: list
    conf: float
    cls: int

    def __post_init__(self) -> None:
        self.xyxy = [np.array(self.xyxy)]
        self.conf = np.array([self.conf])
        self.cls = np.array([self.cls])


class FakeResults:
    """Fake YOLO results container."""

    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


def _make_mock_model(results_per_call: list[list[FakeBox]]):
    """Create a mock YOLO model returning specified results."""
    model = MagicMock()
    side_effects = []
    for boxes in results_per_call:
        side_effects.append([FakeResults(boxes)])
    model.side_effect = side_effects
    return model


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_fields(self) -> None:
        """Detection stores bbox, confidence, class_id, and centroid."""
        det = Detection(
            bbox=(10.0, 20.0, 50.0, 80.0),
            confidence=0.95,
            class_id=0,
            centroid=(30.0, 50.0),
        )
        assert det.bbox == (10.0, 20.0, 50.0, 80.0)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.centroid == (30.0, 50.0)


class TestPersonDetector:
    """Tests for the PersonDetector class."""

    @patch("src.detection.detector.YOLO")
    def test_init_loads_model(self, mock_yolo_cls: MagicMock) -> None:
        """Detector initializes YOLO with the given model path."""
        detector = PersonDetector(model_path="yolov8n.pt", confidence_threshold=0.6)
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        assert detector.confidence_threshold == 0.6

    @patch("src.detection.detector.YOLO")
    def test_detect_persons(self, mock_yolo_cls: MagicMock) -> None:
        """Detector returns only person-class detections above threshold."""
        boxes = [
            FakeBox(xyxy=[10, 20, 50, 80], conf=0.9, cls=0),  # person
            FakeBox(xyxy=[100, 100, 200, 200], conf=0.8, cls=1),  # not person
            FakeBox(xyxy=[60, 70, 90, 120], conf=0.3, cls=0),  # below threshold
        ]
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults(boxes)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(confidence_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].class_id == 0
        assert detections[0].confidence == 0.9

    @patch("src.detection.detector.YOLO")
    def test_detect_computes_centroid(self, mock_yolo_cls: MagicMock) -> None:
        """Detection centroid is correctly computed from bounding box."""
        boxes = [FakeBox(xyxy=[10, 20, 50, 80], conf=0.9, cls=0)]
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults(boxes)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].centroid == (30.0, 50.0)

    @patch("src.detection.detector.YOLO")
    def test_detect_empty_frame(self, mock_yolo_cls: MagicMock) -> None:
        """Detector returns empty list when no persons are found."""
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults([])]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    @patch("src.detection.detector.YOLO")
    def test_detect_batch(self, mock_yolo_cls: MagicMock) -> None:
        """Batch detection returns a list of detection lists."""
        boxes1 = [FakeBox(xyxy=[10, 20, 50, 80], conf=0.9, cls=0)]
        boxes2 = [FakeBox(xyxy=[100, 100, 200, 200], conf=0.85, cls=0)]
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults(boxes1), FakeResults(boxes2)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector()
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        results = detector.detect_batch(frames)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

    @patch("src.detection.detector.YOLO")
    def test_custom_confidence_threshold(self, mock_yolo_cls: MagicMock) -> None:
        """Custom confidence threshold filters detections accordingly."""
        boxes = [
            FakeBox(xyxy=[10, 20, 50, 80], conf=0.7, cls=0),
            FakeBox(xyxy=[60, 70, 90, 120], conf=0.85, cls=0),
        ]
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults(boxes)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(confidence_threshold=0.8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].confidence == 0.85

    @patch("src.detection.detector.YOLO")
    def test_multiple_persons_detected(self, mock_yolo_cls: MagicMock) -> None:
        """Multiple persons in a frame are all detected."""
        boxes = [
            FakeBox(xyxy=[10, 20, 50, 80], conf=0.9, cls=0),
            FakeBox(xyxy=[100, 100, 200, 200], conf=0.85, cls=0),
            FakeBox(xyxy=[300, 300, 400, 400], conf=0.75, cls=0),
        ]
        mock_model = MagicMock()
        mock_model.return_value = [FakeResults(boxes)]
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(confidence_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 3
