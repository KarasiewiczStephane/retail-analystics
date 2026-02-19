"""Video I/O utilities for reading, writing, and inspecting video files.

Provides generators for frame-by-frame reading, metadata extraction,
and a writer class for producing annotated output videos.
"""

import logging
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".mp4", ".avi", ".mov"}


def validate_video_path(path: str) -> Path:
    """Validate that a video file exists and has a supported format.

    Args:
        path: Path to the video file.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported video format '{video_path.suffix}'. "
            f"Supported: {SUPPORTED_FORMATS}"
        )
    return video_path


def read_video_frames(
    video_path: str,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield frames from a video file as ``(frame_id, frame)`` tuples.

    Args:
        video_path: Path to the video file.

    Yields:
        Tuple of ``(frame_id, frame)`` where frame is a BGR numpy array.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_id = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1
    finally:
        cap.release()

    logger.info("Read %d frames from %s", frame_id, video_path)


def get_video_metadata(video_path: str) -> dict:
    """Extract metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with keys ``fps``, ``width``, ``height``, ``total_frames``.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    logger.debug("Video metadata for %s: %s", video_path, metadata)
    return metadata


class VideoWriter:
    """OpenCV-based video writer with context manager support.

    Args:
        output_path: Path for the output video file.
        fps: Frames per second for the output.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string (default ``'mp4v'``).
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        self.frame_count = 0
        logger.info(
            "VideoWriter opened: %s (%dx%d @ %.1f fps)",
            output_path,
            width,
            height,
            fps,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to the output video.

        Args:
            frame: BGR frame as a numpy array.
        """
        self.writer.write(frame)
        self.frame_count += 1

    def release(self) -> None:
        """Release the video writer and finalize the file."""
        self.writer.release()
        logger.info(
            "VideoWriter closed: %s (%d frames written)",
            self.output_path,
            self.frame_count,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
