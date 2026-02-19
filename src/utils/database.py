"""SQLite database layer for detection and zone event storage.

Manages schema creation, detection logging, zone event tracking,
and query methods for the retail analytics pipeline.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS zone_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    person_id INTEGER NOT NULL,
    zone_name TEXT NOT NULL,
    event_type TEXT CHECK(event_type IN ('enter', 'exit')),
    frame_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_detections_video ON detections(video_id);
CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(video_id, person_id);
CREATE INDEX IF NOT EXISTS idx_zone_events_video ON zone_events(video_id);
CREATE INDEX IF NOT EXISTS idx_zone_events_zone ON zone_events(video_id, zone_name);
"""


class Database:
    """SQLite database for storing detections and zone events.

    Args:
        db_path: Path to the SQLite database file. Use ``':memory:'``
            for an in-memory database.
    """

    def __init__(self, db_path: str = "data/detections.db") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.init_schema()
        logger.info("Database initialized at %s", db_path)

    def init_schema(self) -> None:
        """Create database tables and indices if they do not exist."""
        self.cursor.executescript(SCHEMA)
        self.conn.commit()

    @contextmanager
    def transaction(self):
        """Context manager for explicit transaction handling.

        Yields:
            The database cursor.

        Raises:
            Exception: Re-raises after rollback if an error occurs.
        """
        try:
            yield self.cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def log_detection(
        self,
        video_id: str,
        frame_id: int,
        person_id: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
    ) -> None:
        """Log a single person detection to the database.

        Args:
            video_id: Identifier for the source video.
            frame_id: Frame number in the video.
            person_id: Tracked person identifier.
            bbox: Bounding box as ``(x1, y1, x2, y2)``.
            confidence: Detection confidence score.
        """
        x1, y1, x2, y2 = bbox
        self.cursor.execute(
            """
            INSERT INTO detections
                (video_id, frame_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, frame_id, person_id, x1, y1, x2, y2, confidence),
        )
        self.conn.commit()

    def log_detection_batch(self, detections: list[dict]) -> None:
        """Insert multiple detections in a single transaction.

        Args:
            detections: List of dicts with keys ``video_id``, ``frame_id``,
                ``person_id``, ``bbox_x1``, ``bbox_y1``, ``bbox_x2``,
                ``bbox_y2``, ``confidence``.
        """
        self.cursor.executemany(
            """
            INSERT INTO detections
                (video_id, frame_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)
            VALUES
                (:video_id, :frame_id, :person_id, :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :confidence)
            """,
            detections,
        )
        self.conn.commit()

    def log_zone_event(
        self,
        video_id: str,
        person_id: int,
        zone_name: str,
        event_type: str,
        frame_id: int,
    ) -> None:
        """Log a zone entry or exit event.

        Args:
            video_id: Identifier for the source video.
            person_id: Tracked person identifier.
            zone_name: Name of the zone.
            event_type: Either ``'enter'`` or ``'exit'``.
            frame_id: Frame number when the event occurred.
        """
        self.cursor.execute(
            """
            INSERT INTO zone_events (video_id, person_id, zone_name, event_type, frame_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (video_id, person_id, zone_name, event_type, frame_id),
        )
        self.conn.commit()

    def log_entry_event(self, video_id: str, person_id: int, frame_id: int) -> None:
        """Log a person entering the video frame.

        Args:
            video_id: Identifier for the source video.
            person_id: Tracked person identifier.
            frame_id: Frame number when the person entered.
        """
        self.log_zone_event(video_id, person_id, "frame", "enter", frame_id)

    def log_exit_event(self, video_id: str, person_id: int, frame_id: int) -> None:
        """Log a person exiting the video frame.

        Args:
            video_id: Identifier for the source video.
            person_id: Tracked person identifier.
            frame_id: Frame number when the person exited.
        """
        self.log_zone_event(video_id, person_id, "frame", "exit", frame_id)

    def get_detections_by_video(self, video_id: str) -> list[dict]:
        """Retrieve all detections for a given video.

        Args:
            video_id: Identifier for the source video.

        Returns:
            List of detection records as dictionaries.
        """
        self.cursor.execute(
            "SELECT * FROM detections WHERE video_id = ? ORDER BY frame_id",
            (video_id,),
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_person_trajectory(self, video_id: str, person_id: int) -> list[dict]:
        """Retrieve the trajectory (all detections) for a specific person.

        Args:
            video_id: Identifier for the source video.
            person_id: Tracked person identifier.

        Returns:
            List of detection records ordered by frame.
        """
        self.cursor.execute(
            """
            SELECT frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence
            FROM detections
            WHERE video_id = ? AND person_id = ?
            ORDER BY frame_id
            """,
            (video_id, person_id),
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_zone_events(
        self, video_id: str, zone_name: Optional[str] = None
    ) -> list[dict]:
        """Retrieve zone events, optionally filtered by zone name.

        Args:
            video_id: Identifier for the source video.
            zone_name: Optional zone name to filter by.

        Returns:
            List of zone event records as dictionaries.
        """
        if zone_name:
            self.cursor.execute(
                """
                SELECT * FROM zone_events
                WHERE video_id = ? AND zone_name = ?
                ORDER BY frame_id
                """,
                (video_id, zone_name),
            )
        else:
            self.cursor.execute(
                "SELECT * FROM zone_events WHERE video_id = ? ORDER BY frame_id",
                (video_id,),
            )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_unique_person_count(self, video_id: str) -> int:
        """Count unique persons detected in a video.

        Args:
            video_id: Identifier for the source video.

        Returns:
            Number of unique person IDs.
        """
        self.cursor.execute(
            "SELECT COUNT(DISTINCT person_id) FROM detections WHERE video_id = ?",
            (video_id,),
        )
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.info("Database connection closed")
