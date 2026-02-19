"""Tests for video utility functions."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.video_utils import (
    VideoWriter,
    get_video_metadata,
    read_video_frames,
    validate_video_path,
)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a small sample video for testing."""
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for _ in range(10):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestValidateVideoPath:
    """Tests for video path validation."""

    def test_valid_mp4(self, sample_video: Path) -> None:
        """Valid .mp4 file passes validation."""
        result = validate_video_path(str(sample_video))
        assert result == sample_video

    def test_missing_file(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_video_path("/nonexistent/video.mp4")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Unsupported format raises ValueError."""
        bad_file = tmp_path / "video.mkv"
        bad_file.touch()
        with pytest.raises(ValueError, match="Unsupported video format"):
            validate_video_path(str(bad_file))


class TestReadVideoFrames:
    """Tests for the frame reader generator."""

    def test_reads_all_frames(self, sample_video: Path) -> None:
        """Generator yields all frames in the video."""
        frames = list(read_video_frames(str(sample_video)))
        assert len(frames) == 10

    def test_frame_ids_sequential(self, sample_video: Path) -> None:
        """Frame IDs are sequential starting from 0."""
        frames = list(read_video_frames(str(sample_video)))
        ids = [fid for fid, _ in frames]
        assert ids == list(range(10))

    def test_frame_is_numpy_array(self, sample_video: Path) -> None:
        """Yielded frames are numpy arrays with correct shape."""
        for _, frame in read_video_frames(str(sample_video)):
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (240, 320, 3)
            break

    def test_invalid_path_raises(self) -> None:
        """Nonexistent video path raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to open"):
            list(read_video_frames("/nonexistent/video.mp4"))


class TestGetVideoMetadata:
    """Tests for video metadata extraction."""

    def test_metadata_keys(self, sample_video: Path) -> None:
        """Metadata dict contains expected keys."""
        meta = get_video_metadata(str(sample_video))
        assert "fps" in meta
        assert "width" in meta
        assert "height" in meta
        assert "total_frames" in meta

    def test_metadata_values(self, sample_video: Path) -> None:
        """Metadata values match the created video."""
        meta = get_video_metadata(str(sample_video))
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["total_frames"] == 10

    def test_invalid_path_raises(self) -> None:
        """Nonexistent file raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to open"):
            get_video_metadata("/nonexistent/video.mp4")


class TestVideoWriter:
    """Tests for the VideoWriter class."""

    def test_write_frames(self, tmp_path: Path) -> None:
        """Writer produces a valid video file."""
        output = tmp_path / "output.mp4"
        with VideoWriter(str(output), 30.0, 320, 240) as writer:
            for _ in range(5):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                writer.write(frame)
        assert output.exists()
        assert writer.frame_count == 5

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Writer creates parent directories if they do not exist."""
        output = tmp_path / "subdir" / "deep" / "output.mp4"
        writer = VideoWriter(str(output), 30.0, 320, 240)
        writer.release()
        assert output.parent.exists()
