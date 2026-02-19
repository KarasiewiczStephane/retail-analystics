"""Tests for SQLite database operations."""

import pytest

from src.utils.database import Database


@pytest.fixture
def db() -> Database:
    """Create an in-memory database for testing."""
    database = Database(":memory:")
    yield database
    database.close()


class TestDatabaseSchema:
    """Tests for database schema initialization."""

    def test_schema_creation(self, db: Database) -> None:
        """Schema creates detections and zone_events tables."""
        db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in db.cursor.fetchall()}
        assert "detections" in tables
        assert "zone_events" in tables

    def test_schema_idempotent(self, db: Database) -> None:
        """Calling init_schema twice does not raise errors."""
        db.init_schema()
        db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in db.cursor.fetchall()}
        assert "detections" in tables

    def test_indices_created(self, db: Database) -> None:
        """Schema creates expected indices."""
        db.cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row["name"] for row in db.cursor.fetchall()}
        assert "idx_detections_video" in indices
        assert "idx_zone_events_video" in indices


class TestDetectionLogging:
    """Tests for detection insert and query methods."""

    def test_log_single_detection(self, db: Database) -> None:
        """Single detection is logged and retrievable."""
        db.log_detection("vid1", 0, 1, (10.0, 20.0, 50.0, 80.0), 0.95)
        results = db.get_detections_by_video("vid1")
        assert len(results) == 1
        assert results[0]["person_id"] == 1
        assert results[0]["confidence"] == 0.95

    def test_log_detection_batch(self, db: Database) -> None:
        """Batch insert logs multiple detections."""
        batch = [
            {
                "video_id": "vid1",
                "frame_id": 0,
                "person_id": 1,
                "bbox_x1": 10,
                "bbox_y1": 20,
                "bbox_x2": 50,
                "bbox_y2": 80,
                "confidence": 0.9,
            },
            {
                "video_id": "vid1",
                "frame_id": 0,
                "person_id": 2,
                "bbox_x1": 100,
                "bbox_y1": 120,
                "bbox_x2": 150,
                "bbox_y2": 180,
                "confidence": 0.85,
            },
        ]
        db.log_detection_batch(batch)
        results = db.get_detections_by_video("vid1")
        assert len(results) == 2

    def test_get_detections_empty(self, db: Database) -> None:
        """Query for nonexistent video returns empty list."""
        results = db.get_detections_by_video("nonexistent")
        assert results == []

    def test_get_person_trajectory(self, db: Database) -> None:
        """Trajectory query returns ordered detections for a person."""
        for frame_id in range(5):
            db.log_detection(
                "vid1", frame_id, 1, (10.0 + frame_id, 20.0, 50.0, 80.0), 0.9
            )
        trajectory = db.get_person_trajectory("vid1", 1)
        assert len(trajectory) == 5
        assert trajectory[0]["frame_id"] == 0
        assert trajectory[4]["frame_id"] == 4

    def test_unique_person_count(self, db: Database) -> None:
        """Unique person count is computed correctly."""
        db.log_detection("vid1", 0, 1, (10, 20, 50, 80), 0.9)
        db.log_detection("vid1", 1, 1, (15, 25, 55, 85), 0.9)
        db.log_detection("vid1", 0, 2, (100, 120, 150, 180), 0.8)
        assert db.get_unique_person_count("vid1") == 2


class TestZoneEvents:
    """Tests for zone event logging and queries."""

    def test_log_zone_event(self, db: Database) -> None:
        """Zone entry event is logged correctly."""
        db.log_zone_event("vid1", 1, "entrance", "enter", 5)
        events = db.get_zone_events("vid1")
        assert len(events) == 1
        assert events[0]["zone_name"] == "entrance"
        assert events[0]["event_type"] == "enter"

    def test_log_entry_exit_events(self, db: Database) -> None:
        """Entry and exit helper methods log to zone_events."""
        db.log_entry_event("vid1", 1, 0)
        db.log_exit_event("vid1", 1, 10)
        events = db.get_zone_events("vid1")
        assert len(events) == 2
        assert events[0]["event_type"] == "enter"
        assert events[1]["event_type"] == "exit"

    def test_get_zone_events_filtered(self, db: Database) -> None:
        """Zone events can be filtered by zone name."""
        db.log_zone_event("vid1", 1, "entrance", "enter", 0)
        db.log_zone_event("vid1", 1, "checkout", "enter", 5)
        entrance_events = db.get_zone_events("vid1", "entrance")
        assert len(entrance_events) == 1
        assert entrance_events[0]["zone_name"] == "entrance"

    def test_transaction_rollback(self, db: Database) -> None:
        """Transaction context manager rolls back on error."""
        try:
            with db.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO zone_events (video_id, person_id, zone_name, event_type, frame_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("vid1", 1, "entrance", "enter", 0),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass
        events = db.get_zone_events("vid1")
        assert len(events) == 0
