"""Dwell time tracking and zone transition matrix.

Computes how long each person spends in each zone and records
movement patterns between zones as a transition matrix.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DwellRecord:
    """Record of a single dwell period in a zone.

    Attributes:
        person_id: Tracked person identifier.
        zone_name: Name of the zone visited.
        entry_frame: Frame number when the person entered.
        exit_frame: Frame number when the person exited, or None if still inside.
    """

    person_id: int
    zone_name: str
    entry_frame: int
    exit_frame: Optional[int] = None

    def duration_frames(self) -> Optional[int]:
        """Return dwell duration in frames, or None if not yet exited.

        Returns:
            Number of frames spent in the zone, or None.
        """
        if self.exit_frame is None:
            return None
        return self.exit_frame - self.entry_frame

    def duration_seconds(self, fps: float) -> Optional[float]:
        """Return dwell duration in seconds.

        Args:
            fps: Video frame rate for conversion.

        Returns:
            Duration in seconds, or None if not yet exited.
        """
        frames = self.duration_frames()
        if frames is None or fps <= 0:
            return None
        return frames / fps


class DwellTimeTracker:
    """Tracks per-zone dwell times for all persons.

    Args:
        fps: Video frame rate used for time conversion.
    """

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.active_dwells: dict[tuple[int, str], DwellRecord] = {}
        self.completed_dwells: list[DwellRecord] = []

    def start_dwell(self, person_id: int, zone_name: str, frame_id: int) -> None:
        """Begin tracking a dwell period.

        Args:
            person_id: Tracked person identifier.
            zone_name: Name of the zone entered.
            frame_id: Frame number of entry.
        """
        key = (person_id, zone_name)
        self.active_dwells[key] = DwellRecord(
            person_id=person_id,
            zone_name=zone_name,
            entry_frame=frame_id,
        )

    def end_dwell(
        self, person_id: int, zone_name: str, frame_id: int
    ) -> Optional[DwellRecord]:
        """End a dwell period and record the completed duration.

        Args:
            person_id: Tracked person identifier.
            zone_name: Name of the zone exited.
            frame_id: Frame number of exit.

        Returns:
            The completed DwellRecord, or None if no active dwell was found.
        """
        key = (person_id, zone_name)
        if key in self.active_dwells:
            record = self.active_dwells.pop(key)
            record.exit_frame = frame_id
            self.completed_dwells.append(record)
            return record
        return None

    def get_zone_average_dwell_time(self, zone_name: str) -> float:
        """Compute the average dwell time in seconds for a zone.

        Args:
            zone_name: Name of the zone.

        Returns:
            Average dwell time in seconds, or 0.0 if no data.
        """
        durations = self._zone_durations(zone_name)
        return float(np.mean(durations)) if durations else 0.0

    def get_zone_dwell_stats(self, zone_name: str) -> dict[str, float]:
        """Compute dwell time statistics for a zone.

        Args:
            zone_name: Name of the zone.

        Returns:
            Dictionary with ``min``, ``max``, ``avg``, ``median`` keys (seconds).
        """
        durations = self._zone_durations(zone_name)
        if not durations:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0}
        return {
            "min": float(min(durations)),
            "max": float(max(durations)),
            "avg": float(np.mean(durations)),
            "median": float(np.median(durations)),
        }

    def finalize_active_dwells(self, final_frame: int) -> None:
        """Close all active dwell periods at the end of a video.

        Args:
            final_frame: The last frame number of the video.
        """
        for record in self.active_dwells.values():
            record.exit_frame = final_frame
            self.completed_dwells.append(record)
        self.active_dwells.clear()

    def _zone_durations(self, zone_name: str) -> list[float]:
        """Collect completed dwell durations in seconds for a zone."""
        return [
            d.duration_seconds(self.fps)
            for d in self.completed_dwells
            if d.zone_name == zone_name and d.duration_seconds(self.fps) is not None
        ]


class ZoneTransitionMatrix:
    """Records movement patterns between zones.

    Builds a transition count matrix and can normalize it into
    transition probabilities.

    Args:
        zone_names: Ordered list of zone names.
    """

    def __init__(self, zone_names: list[str]) -> None:
        self.zone_names = list(zone_names)
        self.zone_index: dict[str, int] = {name: i for i, name in enumerate(zone_names)}
        n = len(zone_names)
        self.matrix = np.zeros((n, n), dtype=int)
        self.last_zone: dict[int, str] = {}

    def record_zone_exit(self, person_id: int, from_zone: str) -> None:
        """Record that a person exited a zone.

        Args:
            person_id: Tracked person identifier.
            from_zone: Zone name that was exited.
        """
        self.last_zone[person_id] = from_zone

    def record_zone_entry(self, person_id: int, to_zone: str) -> None:
        """Record that a person entered a zone, updating the transition matrix.

        Only counts transitions between different zones.

        Args:
            person_id: Tracked person identifier.
            to_zone: Zone name that was entered.
        """
        if person_id in self.last_zone:
            from_zone = self.last_zone[person_id]
            if from_zone != to_zone:
                from_idx = self.zone_index.get(from_zone)
                to_idx = self.zone_index.get(to_zone)
                if from_idx is not None and to_idx is not None:
                    self.matrix[from_idx, to_idx] += 1

    def get_matrix(self) -> np.ndarray:
        """Return the raw transition count matrix.

        Returns:
            2D numpy array of transition counts.
        """
        return self.matrix.copy()

    def get_transition_probability_matrix(self) -> np.ndarray:
        """Return row-normalized transition probabilities.

        Returns:
            2D numpy array where each row sums to 1 (or 0 if no transitions).
        """
        row_sums = self.matrix.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        return self.matrix / row_sums

    def get_top_transitions(self, n: int = 5) -> list[tuple[str, str, int]]:
        """Return the top N zone transitions by count.

        Args:
            n: Number of transitions to return.

        Returns:
            List of ``(from_zone, to_zone, count)`` tuples sorted descending.
        """
        transitions = []
        for i, from_name in enumerate(self.zone_names):
            for j, to_name in enumerate(self.zone_names):
                if self.matrix[i, j] > 0:
                    transitions.append((from_name, to_name, int(self.matrix[i, j])))
        transitions.sort(key=lambda t: t[2], reverse=True)
        return transitions[:n]
