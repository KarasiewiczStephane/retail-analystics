"""Tests for the VideoProcessor orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.detection.processor import VideoProcessor
from src.detection.tracker import TrackedPerson
from src.utils.database import Database


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a 5-frame sample video."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for _ in range(5):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def db() -> Database:
    """Create an in-memory database."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def mock_tracker():
    """Create a mock PersonTracker."""
    tracker = MagicMock()
    tracker.get_all_trajectories.return_value = {
        1: [(160.0, 120.0), (165.0, 125.0)],
    }
    return tracker


class TestVideoProcessor:
    """Tests for the VideoProcessor class."""

    def test_process_video_basic(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Processor returns correct summary for a simple video."""
        mock_tracker.update.return_value = (
            [
                TrackedPerson(
                    track_id=1,
                    bbox=(10.0, 20.0, 50.0, 80.0),
                    confidence=0.9,
                    centroid=(30.0, 50.0),
                    frame_id=0,
                )
            ],
            {1},
            set(),
        )
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        results = processor.process_video(str(sample_video))

        assert results["video_id"] == "test_video"
        assert results["total_frames"] == 5
        assert results["detection_count"] == 5
        assert results["unique_persons"] == 1
        assert results["width"] == 320
        assert results["height"] == 240

    def test_process_video_custom_id(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Custom video_id is used in results."""
        mock_tracker.update.return_value = ([], set(), set())
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        results = processor.process_video(str(sample_video), video_id="custom_id")

        assert results["video_id"] == "custom_id"

    def test_process_video_logs_detections(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Detections are logged to the database."""
        mock_tracker.update.return_value = (
            [
                TrackedPerson(
                    track_id=1,
                    bbox=(10.0, 20.0, 50.0, 80.0),
                    confidence=0.9,
                    centroid=(30.0, 50.0),
                    frame_id=0,
                )
            ],
            set(),
            set(),
        )
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        processor.process_video(str(sample_video), video_id="vid1")

        detections = db.get_detections_by_video("vid1")
        assert len(detections) == 5

    def test_process_video_logs_entry_events(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Entry events are logged to the database."""
        call_count = [0]

        def side_effect(frame):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    [
                        TrackedPerson(
                            track_id=1,
                            bbox=(10.0, 20.0, 50.0, 80.0),
                            confidence=0.9,
                            centroid=(30.0, 50.0),
                            frame_id=0,
                        )
                    ],
                    {1},
                    set(),
                )
            return ([], set(), set())

        mock_tracker.update.side_effect = side_effect
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        processor.process_video(str(sample_video), video_id="vid1")

        events = db.get_zone_events("vid1")
        enter_events = [e for e in events if e["event_type"] == "enter"]
        assert len(enter_events) == 1

    def test_process_video_logs_exit_events(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Exit events are logged to the database."""
        call_count = [0]

        def side_effect(frame):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    [
                        TrackedPerson(
                            track_id=1,
                            bbox=(10.0, 20.0, 50.0, 80.0),
                            confidence=0.9,
                            centroid=(30.0, 50.0),
                            frame_id=0,
                        )
                    ],
                    {1},
                    set(),
                )
            if call_count[0] == 2:
                return ([], set(), {1})
            return ([], set(), set())

        mock_tracker.update.side_effect = side_effect
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        processor.process_video(str(sample_video), video_id="vid1")

        events = db.get_zone_events("vid1")
        exit_events = [e for e in events if e["event_type"] == "exit"]
        assert len(exit_events) == 1

    def test_process_video_progress_callback(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Progress callback is invoked for each frame."""
        mock_tracker.update.return_value = ([], set(), set())
        mock_tracker.reset.return_value = None

        callback = MagicMock()
        processor = VideoProcessor(db, mock_tracker)
        processor.process_video(str(sample_video), progress_callback=callback)

        assert callback.call_count == 5
        callback.assert_any_call(1, 5)
        callback.assert_any_call(5, 5)

    def test_process_video_empty(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Empty detections across all frames yield zero counts."""
        mock_tracker.update.return_value = ([], set(), set())
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        results = processor.process_video(str(sample_video))

        assert results["detection_count"] == 0
        assert results["unique_persons"] == 0

    def test_process_video_multiple_persons(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Multiple persons are counted correctly."""
        mock_tracker.update.return_value = (
            [
                TrackedPerson(1, (10, 20, 50, 80), 0.9, (30, 50), 0),
                TrackedPerson(2, (100, 100, 200, 200), 0.85, (150, 150), 0),
            ],
            set(),
            set(),
        )
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        results = processor.process_video(str(sample_video))

        assert results["unique_persons"] == 2
        assert results["detection_count"] == 10

    def test_process_video_resets_tracker(
        self, db: Database, mock_tracker: MagicMock, sample_video: Path
    ) -> None:
        """Tracker is reset at the start of processing."""
        mock_tracker.update.return_value = ([], set(), set())
        mock_tracker.reset.return_value = None

        processor = VideoProcessor(db, mock_tracker)
        processor.process_video(str(sample_video))

        mock_tracker.reset.assert_called_once()
