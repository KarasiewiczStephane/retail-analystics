"""Tests for ByteTrack multi-person tracking."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.tracker import PersonTracker, TrackState, TrackedPerson


class FakeBoxes:
    """Fake YOLO tracking result boxes."""

    def __init__(self, boxes_data: list[dict] | None = None) -> None:
        if boxes_data is None:
            self.id = None
            self._boxes = []
        else:
            self.id = np.array([b["id"] for b in boxes_data])
            self._boxes = [
                type(
                    "Box",
                    (),
                    {
                        "xyxy": [np.array(b["xyxy"])],
                        "conf": np.array([b["conf"]]),
                    },
                )()
                for b in boxes_data
            ]

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class FakeTrackResult:
    """Fake tracking result container."""

    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


@pytest.fixture
def mock_tracker():
    """Create a PersonTracker with a mocked YOLO model."""
    with patch("src.detection.tracker.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        tracker = PersonTracker(confidence_threshold=0.5)
        yield tracker, mock_model


class TestTrackedPerson:
    """Tests for the TrackedPerson dataclass."""

    def test_fields(self) -> None:
        """TrackedPerson stores all required fields."""
        tp = TrackedPerson(
            track_id=1,
            bbox=(10.0, 20.0, 50.0, 80.0),
            confidence=0.9,
            centroid=(30.0, 50.0),
            frame_id=0,
        )
        assert tp.track_id == 1
        assert tp.frame_id == 0


class TestTrackState:
    """Tests for the TrackState dataclass."""

    def test_defaults(self) -> None:
        """TrackState has correct default values."""
        ts = TrackState(track_id=1, first_frame=0, last_frame=0)
        assert ts.trajectory == []
        assert ts.is_active is True


class TestPersonTracker:
    """Tests for the PersonTracker class."""

    def test_init(self, mock_tracker) -> None:
        """Tracker initializes with empty state."""
        tracker, _ = mock_tracker
        assert tracker.frame_count == 0
        assert len(tracker.tracks) == 0
        assert len(tracker.active_track_ids) == 0

    def test_update_with_detections(self, mock_tracker) -> None:
        """Tracker returns tracked persons with correct IDs."""
        tracker, mock_model = mock_tracker
        boxes = FakeBoxes(
            [
                {"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9},
                {"id": 2, "xyxy": [100, 100, 200, 200], "conf": 0.85},
            ]
        )
        mock_model.track.return_value = [FakeTrackResult(boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons, entered, exited = tracker.update(frame)

        assert len(persons) == 2
        assert entered == {1, 2}
        assert exited == set()
        assert tracker.frame_count == 1

    def test_update_empty_frame(self, mock_tracker) -> None:
        """Empty frame returns no persons and no events."""
        tracker, mock_model = mock_tracker
        boxes = FakeBoxes(None)
        mock_model.track.return_value = [FakeTrackResult(boxes)]

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons, entered, exited = tracker.update(frame)

        assert len(persons) == 0
        assert entered == set()
        assert exited == set()

    def test_entry_detection(self, mock_tracker) -> None:
        """New person appearing triggers an entry event."""
        tracker, mock_model = mock_tracker

        # Frame 1: person 1
        boxes1 = FakeBoxes([{"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9}])
        mock_model.track.return_value = [FakeTrackResult(boxes1)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, entered1, _ = tracker.update(frame)
        assert 1 in entered1

        # Frame 2: person 1 + person 2 (new entry)
        boxes2 = FakeBoxes(
            [
                {"id": 1, "xyxy": [15, 25, 55, 85], "conf": 0.9},
                {"id": 2, "xyxy": [100, 100, 200, 200], "conf": 0.85},
            ]
        )
        mock_model.track.return_value = [FakeTrackResult(boxes2)]
        _, entered2, _ = tracker.update(frame)
        assert 2 in entered2
        assert 1 not in entered2

    def test_exit_detection(self, mock_tracker) -> None:
        """Person disappearing triggers an exit event."""
        tracker, mock_model = mock_tracker

        # Frame 1: person 1 and 2
        boxes1 = FakeBoxes(
            [
                {"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9},
                {"id": 2, "xyxy": [100, 100, 200, 200], "conf": 0.85},
            ]
        )
        mock_model.track.return_value = [FakeTrackResult(boxes1)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update(frame)

        # Frame 2: only person 1 (person 2 exits)
        boxes2 = FakeBoxes([{"id": 1, "xyxy": [15, 25, 55, 85], "conf": 0.9}])
        mock_model.track.return_value = [FakeTrackResult(boxes2)]
        _, _, exited = tracker.update(frame)
        assert 2 in exited

    def test_trajectory_accumulation(self, mock_tracker) -> None:
        """Trajectories accumulate centroids over frames."""
        tracker, mock_model = mock_tracker
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(3):
            x = 10 + i * 10
            boxes = FakeBoxes([{"id": 1, "xyxy": [x, 20, x + 40, 80], "conf": 0.9}])
            mock_model.track.return_value = [FakeTrackResult(boxes)]
            tracker.update(frame)

        trajectories = tracker.get_all_trajectories()
        assert 1 in trajectories
        assert len(trajectories[1]) == 3

    def test_get_track_state(self, mock_tracker) -> None:
        """Track state is retrievable after detection."""
        tracker, mock_model = mock_tracker
        boxes = FakeBoxes([{"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9}])
        mock_model.track.return_value = [FakeTrackResult(boxes)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update(frame)

        state = tracker.get_track_state(1)
        assert state is not None
        assert state.track_id == 1
        assert state.is_active is True

    def test_get_track_state_missing(self, mock_tracker) -> None:
        """Missing track returns None."""
        tracker, _ = mock_tracker
        assert tracker.get_track_state(999) is None

    def test_reset(self, mock_tracker) -> None:
        """Reset clears all tracker state."""
        tracker, mock_model = mock_tracker
        boxes = FakeBoxes([{"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9}])
        mock_model.track.return_value = [FakeTrackResult(boxes)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.update(frame)

        tracker.reset()
        assert tracker.frame_count == 0
        assert len(tracker.tracks) == 0
        assert len(tracker.active_track_ids) == 0

    def test_track_state_deactivates_on_exit(self, mock_tracker) -> None:
        """Track state is marked inactive when person exits."""
        tracker, mock_model = mock_tracker
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        boxes1 = FakeBoxes([{"id": 1, "xyxy": [10, 20, 50, 80], "conf": 0.9}])
        mock_model.track.return_value = [FakeTrackResult(boxes1)]
        tracker.update(frame)

        boxes2 = FakeBoxes(None)
        mock_model.track.return_value = [FakeTrackResult(boxes2)]
        tracker.update(frame)

        state = tracker.get_track_state(1)
        assert state is not None
        assert state.is_active is False
