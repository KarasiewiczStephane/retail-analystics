"""Zone definition and per-zone traffic analysis.

Manages polygon-based zone definitions, detects zone entry/exit events
for tracked persons, and computes per-zone visitor metrics.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A spatial zone defined by a polygon with a display color.

    Attributes:
        name: Human-readable zone identifier.
        polygon: Shapely Polygon defining the zone boundary.
        color: BGR color tuple for visualization.
    """

    name: str
    polygon: Polygon
    color: tuple[int, int, int] = (0, 255, 0)

    @classmethod
    def from_points(
        cls,
        name: str,
        points: list[list[int]],
        color: tuple[int, int, int] | None = None,
    ) -> "Zone":
        """Create a Zone from a list of vertex coordinates.

        Args:
            name: Zone name.
            points: List of ``[x, y]`` vertex coordinates.
            color: Optional BGR display color.

        Returns:
            Zone instance with a polygon constructed from the given points.
        """
        return cls(
            name=name,
            polygon=Polygon(points),
            color=color or (0, 255, 0),
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check whether a point falls inside this zone.

        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.

        Returns:
            True if the point is inside the polygon.
        """
        return self.polygon.contains(Point(x, y))


@dataclass
class ZoneMetrics:
    """Aggregated visitor metrics for a single zone.

    Attributes:
        zone_name: Name of the zone.
        unique_visitors: Count of distinct person IDs that entered.
        total_entries: Total number of entry events.
        visitor_ids: Set of person IDs that visited.
    """

    zone_name: str
    unique_visitors: int = 0
    total_entries: int = 0
    visitor_ids: set[int] = field(default_factory=set)


class ZoneAnalyzer:
    """Tracks persons across polygon zones and computes per-zone metrics.

    Args:
        zones: List of Zone objects defining the spatial layout.
    """

    def __init__(self, zones: list[Zone]) -> None:
        self.zones: dict[str, Zone] = {z.name: z for z in zones}
        self.person_zone_state: dict[int, set[str]] = defaultdict(set)
        self.zone_metrics: dict[str, ZoneMetrics] = {
            z.name: ZoneMetrics(zone_name=z.name) for z in zones
        }
        self.zone_entries: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.zone_exits: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def update(
        self,
        person_id: int,
        centroid: tuple[float, float],
        frame_id: int,
    ) -> dict[str, str]:
        """Update zone state for a person and return zone events.

        Args:
            person_id: Tracked person identifier.
            centroid: Current position as ``(cx, cy)``.
            frame_id: Current frame number.

        Returns:
            Dictionary mapping zone names to event types
            (``'enter'`` or ``'exit'``).
        """
        cx, cy = centroid
        current_zones: set[str] = set()
        events: dict[str, str] = {}

        for zone_name, zone in self.zones.items():
            if zone.contains_point(cx, cy):
                current_zones.add(zone_name)

        entered = current_zones - self.person_zone_state[person_id]
        for zone_name in entered:
            events[zone_name] = "enter"
            self.zone_metrics[zone_name].visitor_ids.add(person_id)
            self.zone_metrics[zone_name].total_entries += 1
            self.zone_entries[zone_name].append((person_id, frame_id))

        exited = self.person_zone_state[person_id] - current_zones
        for zone_name in exited:
            events[zone_name] = "exit"
            self.zone_exits[zone_name].append((person_id, frame_id))

        self.person_zone_state[person_id] = current_zones
        return events

    def get_current_zones(self, person_id: int) -> set[str]:
        """Return the set of zones a person is currently in.

        Args:
            person_id: Tracked person identifier.

        Returns:
            Set of zone names.
        """
        return self.person_zone_state.get(person_id, set())

    def get_zone_metrics(self, zone_name: str) -> ZoneMetrics:
        """Retrieve metrics for a specific zone.

        Args:
            zone_name: Name of the zone.

        Returns:
            ZoneMetrics with updated unique_visitors count.

        Raises:
            KeyError: If the zone name is not found.
        """
        metrics = self.zone_metrics[zone_name]
        metrics.unique_visitors = len(metrics.visitor_ids)
        return metrics

    def get_all_metrics(self) -> dict[str, ZoneMetrics]:
        """Retrieve metrics for all zones.

        Returns:
            Dictionary mapping zone names to their ZoneMetrics.
        """
        for name in self.zone_metrics:
            self.zone_metrics[name].unique_visitors = len(
                self.zone_metrics[name].visitor_ids
            )
        return dict(self.zone_metrics)

    @classmethod
    def from_config(cls, zones_config: list[dict]) -> "ZoneAnalyzer":
        """Create a ZoneAnalyzer from a list of zone configuration dicts.

        Args:
            zones_config: List of dicts with keys ``name``, ``polygon``,
                and optional ``color``.

        Returns:
            Configured ZoneAnalyzer instance.
        """
        zones = [
            Zone.from_points(
                z["name"],
                z["polygon"],
                tuple(z["color"]) if "color" in z else None,
            )
            for z in zones_config
        ]
        logger.info("ZoneAnalyzer created with %d zones", len(zones))
        return cls(zones)
