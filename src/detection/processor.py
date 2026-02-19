"""Video processing orchestrator for detection, tracking, and logging.

Coordinates the full pipeline: reads video frames, runs person tracking,
logs detections and entry/exit events to the database, and returns summary
statistics.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..utils.database import Database
from ..utils.video_utils import get_video_metadata, read_video_frames
from .tracker import PersonTracker

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Orchestrates video processing with detection, tracking, and logging.

    Args:
        db: Database instance for persisting detections and events.
        tracker: PersonTracker instance for multi-person tracking.
    """

    def __init__(
        self,
        db: Database,
        tracker: PersonTracker,
    ) -> None:
        self.db = db
        self.tracker = tracker

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Any]:
        """Process an entire video: detect, track, and log results.

        Args:
            video_path: Path to the input video file.
            video_id: Identifier for this video in the database.
                Defaults to the file stem.
            progress_callback: Optional ``callback(current_frame, total_frames)``
                invoked after each frame for progress reporting.

        Returns:
            Dictionary with keys ``video_id``, ``total_frames``,
            ``detection_count``, ``unique_persons``, ``trajectories``,
            ``width``, ``height``, ``fps``.
        """
        video_id = video_id or Path(video_path).stem
        metadata = get_video_metadata(video_path)
        total_frames = metadata["total_frames"]

        self.tracker.reset()
        detection_count = 0
        unique_persons: set[int] = set()
        batch: list[dict] = []

        logger.info(
            "Processing video '%s': %d frames at %.1f fps (%dx%d)",
            video_id,
            total_frames,
            metadata["fps"],
            metadata["width"],
            metadata["height"],
        )

        for frame_id, frame in read_video_frames(video_path):
            tracked_persons, entries, exits = self.tracker.update(frame)

            for person in tracked_persons:
                batch.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "person_id": person.track_id,
                        "bbox_x1": person.bbox[0],
                        "bbox_y1": person.bbox[1],
                        "bbox_x2": person.bbox[2],
                        "bbox_y2": person.bbox[3],
                        "confidence": person.confidence,
                    }
                )

            if len(batch) >= 100:
                self.db.log_detection_batch(batch)
                batch.clear()

            for person_id in entries:
                self.db.log_entry_event(video_id, person_id, frame_id)
            for person_id in exits:
                self.db.log_exit_event(video_id, person_id, frame_id)

            detection_count += len(tracked_persons)
            unique_persons.update(p.track_id for p in tracked_persons)

            if progress_callback:
                progress_callback(frame_id + 1, total_frames)

        if batch:
            self.db.log_detection_batch(batch)

        results = {
            "video_id": video_id,
            "total_frames": total_frames,
            "detection_count": detection_count,
            "unique_persons": len(unique_persons),
            "trajectories": self.tracker.get_all_trajectories(),
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
        }

        logger.info(
            "Video '%s' processed: %d detections, %d unique persons",
            video_id,
            detection_count,
            len(unique_persons),
        )

        return results
