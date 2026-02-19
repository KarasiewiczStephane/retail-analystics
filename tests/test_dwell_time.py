"""Tests for dwell time tracking and zone transition matrix."""

import pytest

from src.analytics.dwell_time import (
    DwellRecord,
    DwellTimeTracker,
    ZoneTransitionMatrix,
)


class TestDwellRecord:
    """Tests for the DwellRecord dataclass."""

    def test_duration_frames(self) -> None:
        """Duration in frames is computed correctly."""
        record = DwellRecord(person_id=1, zone_name="a", entry_frame=10, exit_frame=40)
        assert record.duration_frames() == 30

    def test_duration_frames_none(self) -> None:
        """Duration is None when exit_frame is not set."""
        record = DwellRecord(person_id=1, zone_name="a", entry_frame=10)
        assert record.duration_frames() is None

    def test_duration_seconds(self) -> None:
        """Duration in seconds uses fps conversion."""
        record = DwellRecord(person_id=1, zone_name="a", entry_frame=0, exit_frame=60)
        assert record.duration_seconds(30.0) == 2.0

    def test_duration_seconds_zero_fps(self) -> None:
        """Zero FPS returns None."""
        record = DwellRecord(person_id=1, zone_name="a", entry_frame=0, exit_frame=60)
        assert record.duration_seconds(0.0) is None


class TestDwellTimeTracker:
    """Tests for the DwellTimeTracker class."""

    def test_start_and_end_dwell(self) -> None:
        """Start and end dwell produces a completed record."""
        tracker = DwellTimeTracker(fps=30.0)
        tracker.start_dwell(1, "entrance", 0)
        record = tracker.end_dwell(1, "entrance", 60)
        assert record is not None
        assert record.duration_seconds(30.0) == 2.0

    def test_end_dwell_missing(self) -> None:
        """Ending a non-existent dwell returns None."""
        tracker = DwellTimeTracker(fps=30.0)
        assert tracker.end_dwell(1, "entrance", 10) is None

    def test_zone_average_dwell_time(self) -> None:
        """Average dwell time is computed correctly."""
        tracker = DwellTimeTracker(fps=30.0)
        tracker.start_dwell(1, "a", 0)
        tracker.end_dwell(1, "a", 30)
        tracker.start_dwell(2, "a", 0)
        tracker.end_dwell(2, "a", 90)
        avg = tracker.get_zone_average_dwell_time("a")
        assert avg == pytest.approx(2.0)

    def test_zone_average_empty(self) -> None:
        """Average dwell time is 0 for empty zone."""
        tracker = DwellTimeTracker(fps=30.0)
        assert tracker.get_zone_average_dwell_time("nothing") == 0.0

    def test_zone_dwell_stats(self) -> None:
        """Dwell stats include min, max, avg, median."""
        tracker = DwellTimeTracker(fps=30.0)
        tracker.start_dwell(1, "a", 0)
        tracker.end_dwell(1, "a", 30)
        tracker.start_dwell(2, "a", 0)
        tracker.end_dwell(2, "a", 90)
        tracker.start_dwell(3, "a", 0)
        tracker.end_dwell(3, "a", 60)

        stats = tracker.get_zone_dwell_stats("a")
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["avg"] == pytest.approx(2.0)
        assert stats["median"] == pytest.approx(2.0)

    def test_zone_dwell_stats_empty(self) -> None:
        """Empty zone returns zero stats."""
        tracker = DwellTimeTracker(fps=30.0)
        stats = tracker.get_zone_dwell_stats("empty")
        assert stats == {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0}

    def test_finalize_active_dwells(self) -> None:
        """Finalize closes all active dwells at the given frame."""
        tracker = DwellTimeTracker(fps=30.0)
        tracker.start_dwell(1, "a", 0)
        tracker.start_dwell(2, "b", 10)
        tracker.finalize_active_dwells(100)

        assert len(tracker.completed_dwells) == 2
        assert len(tracker.active_dwells) == 0
        assert tracker.completed_dwells[0].exit_frame == 100

    def test_multiple_dwells_same_person(self) -> None:
        """Same person can have multiple dwell periods in the same zone."""
        tracker = DwellTimeTracker(fps=30.0)
        tracker.start_dwell(1, "a", 0)
        tracker.end_dwell(1, "a", 30)
        tracker.start_dwell(1, "a", 60)
        tracker.end_dwell(1, "a", 90)
        assert len(tracker.completed_dwells) == 2


class TestZoneTransitionMatrix:
    """Tests for the ZoneTransitionMatrix class."""

    @pytest.fixture
    def matrix(self) -> ZoneTransitionMatrix:
        """Create a matrix with three zones."""
        return ZoneTransitionMatrix(["entrance", "electronics", "checkout"])

    def test_transition_recorded(self, matrix: ZoneTransitionMatrix) -> None:
        """Transition from entrance to electronics is recorded."""
        matrix.record_zone_exit(1, "entrance")
        matrix.record_zone_entry(1, "electronics")
        m = matrix.get_matrix()
        assert m[0, 1] == 1

    def test_same_zone_not_counted(self, matrix: ZoneTransitionMatrix) -> None:
        """Re-entry to the same zone is not counted as a transition."""
        matrix.record_zone_exit(1, "entrance")
        matrix.record_zone_entry(1, "entrance")
        m = matrix.get_matrix()
        assert m.sum() == 0

    def test_no_previous_zone(self, matrix: ZoneTransitionMatrix) -> None:
        """Entry without prior exit does not record a transition."""
        matrix.record_zone_entry(1, "electronics")
        m = matrix.get_matrix()
        assert m.sum() == 0

    def test_probability_matrix(self, matrix: ZoneTransitionMatrix) -> None:
        """Probability matrix rows sum to 1."""
        matrix.record_zone_exit(1, "entrance")
        matrix.record_zone_entry(1, "electronics")
        matrix.record_zone_exit(2, "entrance")
        matrix.record_zone_entry(2, "checkout")
        prob = matrix.get_transition_probability_matrix()
        assert prob[0].sum() == pytest.approx(1.0)

    def test_probability_matrix_zero_row(self, matrix: ZoneTransitionMatrix) -> None:
        """Rows with no transitions stay at 0."""
        prob = matrix.get_transition_probability_matrix()
        assert prob[2].sum() == pytest.approx(0.0)

    def test_get_top_transitions(self, matrix: ZoneTransitionMatrix) -> None:
        """Top transitions are sorted by count."""
        matrix.record_zone_exit(1, "entrance")
        matrix.record_zone_entry(1, "electronics")
        matrix.record_zone_exit(2, "entrance")
        matrix.record_zone_entry(2, "electronics")
        matrix.record_zone_exit(3, "entrance")
        matrix.record_zone_entry(3, "checkout")

        top = matrix.get_top_transitions(2)
        assert len(top) == 2
        assert top[0] == ("entrance", "electronics", 2)
        assert top[1] == ("entrance", "checkout", 1)

    def test_get_top_transitions_empty(self, matrix: ZoneTransitionMatrix) -> None:
        """Empty matrix returns no transitions."""
        assert matrix.get_top_transitions() == []

    def test_multiple_persons(self, matrix: ZoneTransitionMatrix) -> None:
        """Multiple persons contribute independently to the matrix."""
        for pid in range(1, 4):
            matrix.record_zone_exit(pid, "entrance")
            matrix.record_zone_entry(pid, "checkout")
        m = matrix.get_matrix()
        assert m[0, 2] == 3
