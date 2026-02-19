"""ByteTrack-based multi-person tracking.

Wraps the ultralytics tracking API to maintain consistent person IDs
across video frames, detect entry/exit events, and accumulate trajectories.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

from .detector import _to_numpy

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    """A detected person with a persistent track ID.

    Attributes:
        track_id: Unique identifier maintained across frames.
        bbox: Bounding box coordinates as ``(x1, y1, x2, y2)``.
        confidence: Detection confidence score.
        centroid: Center point of the bounding box.
        frame_id: Frame number where this detection occurred.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    centroid: tuple[float, float]
    frame_id: int


@dataclass
class TrackState:
    """Persistent state for a tracked person across frames.

    Attributes:
        track_id: Unique track identifier.
        first_frame: Frame number when the track first appeared.
        last_frame: Frame number of the most recent detection.
        trajectory: Ordered list of centroid positions.
        is_active: Whether the track is currently visible.
    """

    track_id: int
    first_frame: int
    last_frame: int
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    is_active: bool = True


class PersonTracker:
    """Multi-person tracker using YOLOv8 with ByteTrack.

    Processes video frames sequentially, maintaining consistent person IDs
    and detecting entry/exit events.

    Args:
        model_path: Path to the YOLOv8 model weights.
        confidence_threshold: Minimum detection confidence.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracks: dict[int, TrackState] = {}
        self.active_track_ids: set[int] = set()
        self.frame_count: int = 0
        logger.info(
            "PersonTracker initialized: model=%s, threshold=%.2f",
            model_path,
            confidence_threshold,
        )

    def update(
        self, frame: np.ndarray
    ) -> tuple[list[TrackedPerson], set[int], set[int]]:
        """Process a frame and return tracked persons with entry/exit events.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Tuple of ``(tracked_persons, entered_ids, exited_ids)`` where
            entered_ids are new tracks and exited_ids are tracks that
            disappeared in this frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence_threshold,
            verbose=False,
        )[0]

        current_ids: set[int] = set()
        tracked_persons: list[TrackedPerson] = []

        if results.boxes.id is not None:
            ids = _to_numpy(results.boxes.id)
            for i, box in enumerate(results.boxes):
                track_id = int(ids[i])
                coords = _to_numpy(box.xyxy[0])
                x1, y1, x2, y2 = coords
                conf = float(_to_numpy(box.conf).flat[0])
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                current_ids.add(track_id)

                if track_id not in self.tracks:
                    self.tracks[track_id] = TrackState(
                        track_id=track_id,
                        first_frame=self.frame_count,
                        last_frame=self.frame_count,
                    )

                self.tracks[track_id].last_frame = self.frame_count
                self.tracks[track_id].trajectory.append((cx, cy))
                self.tracks[track_id].is_active = True

                tracked_persons.append(
                    TrackedPerson(
                        track_id=track_id,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                        centroid=(cx, cy),
                        frame_id=self.frame_count,
                    )
                )

        exited_ids = self.active_track_ids - current_ids
        for track_id in exited_ids:
            if track_id in self.tracks:
                self.tracks[track_id].is_active = False

        entered_ids = current_ids - self.active_track_ids

        self.active_track_ids = current_ids
        self.frame_count += 1

        return tracked_persons, entered_ids, exited_ids

    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        """Retrieve the state of a specific track.

        Args:
            track_id: The track identifier to look up.

        Returns:
            TrackState if found, otherwise None.
        """
        return self.tracks.get(track_id)

    def get_all_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Retrieve all accumulated trajectories.

        Returns:
            Dictionary mapping track IDs to lists of centroid positions.
        """
        return {tid: ts.trajectory for tid, ts in self.tracks.items()}

    def reset(self) -> None:
        """Reset tracker state for processing a new video."""
        self.tracks.clear()
        self.active_track_ids.clear()
        self.frame_count = 0
        logger.info("PersonTracker reset")
