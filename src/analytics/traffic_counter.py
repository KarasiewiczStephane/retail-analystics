"""Foot traffic counting with hourly aggregation.

Tracks visitor entries per frame and aggregates counts into hourly
distribution buckets for peak-hour analysis.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrafficCounter:
    """Counts visitor traffic and computes hourly distributions.

    Args:
        fps: Video frame rate for time-based aggregation.
    """

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.hourly_counts: dict[int, int] = defaultdict(int)
        self.frame_counts: dict[int, int] = defaultdict(int)
        self._total_entries: int = 0

    def record_entry(self, frame_id: int) -> None:
        """Record a person entry at the given frame.

        Args:
            frame_id: Frame number when the person entered.
        """
        self.frame_counts[frame_id] += 1
        hour = int(frame_id / self.fps / 3600) if self.fps > 0 else 0
        self.hourly_counts[hour] += 1
        self._total_entries += 1

    def get_hourly_distribution(self) -> dict[int, int]:
        """Return visitor counts aggregated by hour.

        Returns:
            Dictionary mapping hour index to visitor count.
        """
        return dict(self.hourly_counts)

    def get_peak_hour(self) -> int | None:
        """Return the hour with the highest visitor count.

        Returns:
            Hour index with the most traffic, or None if no data.
        """
        if not self.hourly_counts:
            return None
        return max(self.hourly_counts, key=self.hourly_counts.get)

    def get_total_visitors(self) -> int:
        """Return the total number of recorded entries.

        Returns:
            Total entry count.
        """
        return self._total_entries
