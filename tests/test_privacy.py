"""Tests for the privacy pipeline (face detection and blurring)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.detection.privacy import FaceBlurrer, FaceDetection, HaarFaceBlurrer


class TestFaceDetection:
    """Tests for the FaceDetection dataclass."""

    def test_fields(self) -> None:
        """FaceDetection stores bbox and confidence."""
        fd = FaceDetection(bbox=(10, 20, 50, 80), confidence=0.95)
        assert fd.bbox == (10, 20, 50, 80)
        assert fd.confidence == 0.95


class TestHaarFaceBlurrer:
    """Tests for the Haar cascade face blurrer."""

    def test_blur_intensity_forced_odd(self) -> None:
        """Even blur intensity is rounded up to odd."""
        blurrer = HaarFaceBlurrer(blur_intensity=50)
        assert blurrer.blur_intensity == 51

    def test_blur_intensity_already_odd(self) -> None:
        """Odd blur intensity is kept as-is."""
        blurrer = HaarFaceBlurrer(blur_intensity=51)
        assert blurrer.blur_intensity == 51

    def test_blur_face_modifies_region(self) -> None:
        """Blurring changes pixel values in the face region."""
        blurrer = HaarFaceBlurrer(blur_intensity=11)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        blurrer.blur_face(frame, (20, 20, 80, 80))
        region = frame[20:80, 20:80]
        original_region = original[20:80, 20:80]
        assert not np.array_equal(region, original_region)

    def test_blur_face_boundary_clipping(self) -> None:
        """Face bbox extending beyond frame is clipped."""
        blurrer = HaarFaceBlurrer(blur_intensity=11)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = blurrer.blur_face(frame, (-10, -10, 50, 50))
        assert result.shape == (100, 100, 3)

    def test_blur_face_invalid_bbox(self) -> None:
        """Zero-area bbox does not crash."""
        blurrer = HaarFaceBlurrer(blur_intensity=11)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = blurrer.blur_face(frame, (50, 50, 50, 50))
        assert result.shape == (100, 100, 3)

    def test_process_frame_returns_copy(self) -> None:
        """process_frame returns a copy, not modifying original."""
        blurrer = HaarFaceBlurrer(blur_intensity=11)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = blurrer.process_frame(frame)
        assert result is not frame


class TestFaceBlurrer:
    """Tests for the YOLO-based face blurrer with fallback."""

    def test_fallback_to_haar(self) -> None:
        """When YOLO model fails to load, Haar cascade is used."""
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer(model_path="nonexistent.pt")
            assert blurrer._use_yolo is False

    def test_blur_intensity_forced_odd(self) -> None:
        """Even blur intensity is rounded up to odd."""
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer(blur_intensity=50)
            assert blurrer.blur_intensity == 51

    def test_process_frame_with_fallback(self) -> None:
        """process_frame works with Haar fallback."""
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = blurrer.process_frame(frame)
            assert result.shape == frame.shape

    def test_blur_face_delegates(self) -> None:
        """blur_face delegates to the fallback blurrer."""
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer(blur_intensity=11)
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            original = frame.copy()
            blurrer.blur_face(frame, (20, 20, 80, 80))
            assert not np.array_equal(frame[20:80, 20:80], original[20:80, 20:80])

    def test_detect_faces_with_haar_fallback(self) -> None:
        """detect_faces returns results using Haar cascade."""
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            faces = blurrer.detect_faces(frame)
            assert isinstance(faces, list)

    def test_process_video(self, tmp_path: Path) -> None:
        """process_video produces an output video."""
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_path), fourcc, 30.0, (320, 240))
        for _ in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer()
            result = blurrer.process_video(str(input_path), str(output_path))

        assert output_path.exists()
        assert result["frames_processed"] == 5

    def test_process_video_progress_callback(self, tmp_path: Path) -> None:
        """Progress callback is invoked during video processing."""
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_path), fourcc, 30.0, (320, 240))
        for _ in range(3):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        callback = MagicMock()
        with patch("src.detection.privacy._YOLO", None):
            blurrer = FaceBlurrer()
            blurrer.process_video(
                str(input_path), str(output_path), progress_callback=callback
            )

        assert callback.call_count == 3
