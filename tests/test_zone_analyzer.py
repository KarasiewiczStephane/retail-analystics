"""Tests for zone analysis and traffic counting."""

import pytest

from src.analytics.zone_analyzer import Zone, ZoneAnalyzer
from src.analytics.traffic_counter import TrafficCounter


class TestZone:
    """Tests for the Zone class."""

    def test_from_points(self) -> None:
        """Zone is created from a list of points."""
        zone = Zone.from_points("test", [[0, 0], [100, 0], [100, 100], [0, 100]])
        assert zone.name == "test"
        assert zone.contains_point(50, 50)

    def test_contains_point_inside(self) -> None:
        """Point inside the polygon returns True."""
        zone = Zone.from_points("box", [[0, 0], [100, 0], [100, 100], [0, 100]])
        assert zone.contains_point(50, 50)

    def test_contains_point_outside(self) -> None:
        """Point outside the polygon returns False."""
        zone = Zone.from_points("box", [[0, 0], [100, 0], [100, 100], [0, 100]])
        assert not zone.contains_point(200, 200)

    def test_default_color(self) -> None:
        """Default color is green."""
        zone = Zone.from_points("test", [[0, 0], [1, 0], [1, 1], [0, 1]])
        assert zone.color == (0, 255, 0)

    def test_custom_color(self) -> None:
        """Custom color is applied."""
        zone = Zone.from_points(
            "test", [[0, 0], [1, 0], [1, 1], [0, 1]], color=(255, 0, 0)
        )
        assert zone.color == (255, 0, 0)


class TestZoneAnalyzer:
    """Tests for the ZoneAnalyzer class."""

    @pytest.fixture
    def analyzer(self) -> ZoneAnalyzer:
        """Create a ZoneAnalyzer with two non-overlapping zones."""
        zones = [
            Zone.from_points("left", [[0, 0], [50, 0], [50, 100], [0, 100]]),
            Zone.from_points("right", [[60, 0], [110, 0], [110, 100], [60, 100]]),
        ]
        return ZoneAnalyzer(zones)

    @pytest.fixture
    def overlapping_analyzer(self) -> ZoneAnalyzer:
        """Create a ZoneAnalyzer with overlapping zones."""
        zones = [
            Zone.from_points("a", [[0, 0], [100, 0], [100, 100], [0, 100]]),
            Zone.from_points("b", [[50, 0], [150, 0], [150, 100], [50, 100]]),
        ]
        return ZoneAnalyzer(zones)

    def test_enter_zone(self, analyzer: ZoneAnalyzer) -> None:
        """Person entering a zone triggers an enter event."""
        events = analyzer.update(1, (25.0, 50.0), 0)
        assert events == {"left": "enter"}

    def test_exit_zone(self, analyzer: ZoneAnalyzer) -> None:
        """Person leaving a zone triggers an exit event."""
        analyzer.update(1, (25.0, 50.0), 0)
        events = analyzer.update(1, (200.0, 50.0), 1)
        assert events == {"left": "exit"}

    def test_zone_transition(self, analyzer: ZoneAnalyzer) -> None:
        """Moving from one zone to another triggers exit + enter."""
        analyzer.update(1, (25.0, 50.0), 0)
        events = analyzer.update(1, (85.0, 50.0), 1)
        assert events == {"left": "exit", "right": "enter"}

    def test_overlapping_zones(self, overlapping_analyzer: ZoneAnalyzer) -> None:
        """Person in overlapping area triggers entry for both zones."""
        events = overlapping_analyzer.update(1, (75.0, 50.0), 0)
        assert "a" in events
        assert "b" in events

    def test_unique_visitors(self, analyzer: ZoneAnalyzer) -> None:
        """Unique visitor count is correct per zone."""
        analyzer.update(1, (25.0, 50.0), 0)
        analyzer.update(2, (25.0, 50.0), 0)
        analyzer.update(1, (25.0, 50.0), 1)  # same person revisit
        metrics = analyzer.get_zone_metrics("left")
        assert metrics.unique_visitors == 2

    def test_total_entries(self, analyzer: ZoneAnalyzer) -> None:
        """Total entries counts each entry event."""
        analyzer.update(1, (25.0, 50.0), 0)
        analyzer.update(1, (200.0, 50.0), 1)  # exit
        analyzer.update(1, (25.0, 50.0), 2)  # re-enter
        metrics = analyzer.get_zone_metrics("left")
        assert metrics.total_entries == 2

    def test_get_current_zones(self, analyzer: ZoneAnalyzer) -> None:
        """Current zones for a person are tracked correctly."""
        analyzer.update(1, (25.0, 50.0), 0)
        assert analyzer.get_current_zones(1) == {"left"}

    def test_get_current_zones_empty(self, analyzer: ZoneAnalyzer) -> None:
        """Unknown person returns empty set."""
        assert analyzer.get_current_zones(999) == set()

    def test_get_all_metrics(self, analyzer: ZoneAnalyzer) -> None:
        """All metrics returns metrics for every zone."""
        analyzer.update(1, (25.0, 50.0), 0)
        all_metrics = analyzer.get_all_metrics()
        assert "left" in all_metrics
        assert "right" in all_metrics
        assert all_metrics["left"].unique_visitors == 1
        assert all_metrics["right"].unique_visitors == 0

    def test_from_config(self) -> None:
        """ZoneAnalyzer is created from config dicts."""
        config = [
            {
                "name": "entrance",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "color": [0, 255, 0],
            },
            {
                "name": "checkout",
                "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]],
            },
        ]
        analyzer = ZoneAnalyzer.from_config(config)
        assert "entrance" in analyzer.zones
        assert "checkout" in analyzer.zones

    def test_no_event_when_staying_in_zone(self, analyzer: ZoneAnalyzer) -> None:
        """No events when person stays in the same zone."""
        analyzer.update(1, (25.0, 50.0), 0)
        events = analyzer.update(1, (30.0, 55.0), 1)
        assert events == {}


class TestTrafficCounter:
    """Tests for the TrafficCounter class."""

    def test_record_entry(self) -> None:
        """Entry is recorded and total incremented."""
        counter = TrafficCounter(fps=30.0)
        counter.record_entry(0)
        assert counter.get_total_visitors() == 1

    def test_hourly_distribution(self) -> None:
        """Entries are bucketed into correct hours."""
        counter = TrafficCounter(fps=1.0)
        counter.record_entry(0)  # hour 0
        counter.record_entry(3600)  # hour 1
        counter.record_entry(3601)  # hour 1
        dist = counter.get_hourly_distribution()
        assert dist[0] == 1
        assert dist[1] == 2

    def test_peak_hour(self) -> None:
        """Peak hour is the one with the most entries."""
        counter = TrafficCounter(fps=1.0)
        counter.record_entry(0)
        counter.record_entry(3600)
        counter.record_entry(3601)
        assert counter.get_peak_hour() == 1

    def test_peak_hour_empty(self) -> None:
        """Peak hour returns None with no data."""
        counter = TrafficCounter(fps=30.0)
        assert counter.get_peak_hour() is None

    def test_total_visitors(self) -> None:
        """Total visitor count matches entries."""
        counter = TrafficCounter(fps=30.0)
        for i in range(10):
            counter.record_entry(i)
        assert counter.get_total_visitors() == 10

    def test_zero_fps_handling(self) -> None:
        """Zero FPS does not crash hourly bucketing."""
        counter = TrafficCounter(fps=0.0)
        counter.record_entry(100)
        assert counter.get_total_visitors() == 1
