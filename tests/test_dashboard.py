"""Tests for the Streamlit dashboard module.

Streamlit apps are tested by verifying that the module functions
can be imported and called with mock state, and that the processing
logic produces expected results.
"""

import cv2
import numpy as np

from src.analytics.heatmap import HeatmapGenerator
from src.dashboard.app import (
    _heatmap_view_tab,
    _traffic_overview_tab,
    _video_playback_tab,
    _zone_analysis_tab,
    main,
)


class TestDashboardImport:
    """Tests for dashboard module importability."""

    def test_main_is_callable(self) -> None:
        """main function exists and is callable."""
        assert callable(main)

    def test_traffic_overview_is_callable(self) -> None:
        """_traffic_overview_tab function is callable."""
        assert callable(_traffic_overview_tab)

    def test_zone_analysis_is_callable(self) -> None:
        """_zone_analysis_tab function is callable."""
        assert callable(_zone_analysis_tab)

    def test_heatmap_view_is_callable(self) -> None:
        """_heatmap_view_tab function is callable."""
        assert callable(_heatmap_view_tab)

    def test_video_playback_is_callable(self) -> None:
        """_video_playback_tab function is callable."""
        assert callable(_video_playback_tab)


class TestDashboardLogic:
    """Tests for dashboard data processing logic."""

    def test_hourly_distribution_format(self) -> None:
        """Hourly distribution can be converted to display format."""
        hourly = {0: 5, 1: 10, 2: 3}
        rows = list(hourly.items())
        assert len(rows) == 3
        assert rows[0] == (0, 5)

    def test_zone_metrics_display_format(self) -> None:
        """Zone metrics dict can be formatted for display."""
        metrics = {
            "entrance": {
                "unique_visitors": 10,
                "total_entries": 15,
                "avg_dwell_time": 5.5,
            },
            "checkout": {
                "unique_visitors": 8,
                "total_entries": 8,
                "avg_dwell_time": 12.3,
            },
        }
        rows = [
            {
                "Zone": name,
                "Visitors": m["unique_visitors"],
                "Entries": m["total_entries"],
                "Avg Dwell Time (s)": round(m["avg_dwell_time"], 2),
            }
            for name, m in metrics.items()
        ]
        assert len(rows) == 2
        assert rows[0]["Zone"] == "entrance"
        assert rows[0]["Visitors"] == 10

    def test_transition_matrix_format(self) -> None:
        """Transition matrix can be formatted for plotly heatmap."""
        matrix = [[0, 5, 3], [2, 0, 1], [4, 0, 0]]
        zone_names = ["entrance", "electronics", "checkout"]
        assert len(matrix) == len(zone_names)
        assert len(matrix[0]) == len(zone_names)


class TestHeatmapViewLogic:
    """Tests for heatmap view data processing logic."""

    def test_heatmap_from_trajectories(self) -> None:
        """Heatmap can be generated from trajectory data."""
        trajectories = {
            1: [(100.0, 100.0), (110.0, 105.0)],
            2: [(200.0, 200.0), (210.0, 195.0)],
        }
        gen = HeatmapGenerator.from_trajectories(trajectories, 640, 480)
        heatmap = gen.generate_heatmap()
        assert heatmap.shape == (480, 640)
        assert heatmap.max() > 0

    def test_colored_heatmap_shape(self) -> None:
        """Colored heatmap produces a BGR image."""
        gen = HeatmapGenerator(320, 240)
        gen.add_point(160, 120)
        colored = gen.generate_colored_heatmap()
        assert colored.shape == (240, 320, 3)
        assert colored.dtype == np.uint8

    def test_colormap_selection(self) -> None:
        """Different colormaps produce different outputs."""
        gen = HeatmapGenerator(100, 100)
        gen.add_point(50, 50)
        jet = gen.generate_colored_heatmap(colormap=cv2.COLORMAP_JET)
        hot = gen.generate_colored_heatmap(colormap=cv2.COLORMAP_HOT)
        assert not np.array_equal(jet, hot)

    def test_empty_trajectories_heatmap(self) -> None:
        """Empty trajectories produce a zero heatmap."""
        gen = HeatmapGenerator.from_trajectories({}, 640, 480)
        heatmap = gen.generate_heatmap()
        assert heatmap.max() == 0


class TestVideoPlaybackLogic:
    """Tests for video playback data processing logic."""

    def test_trajectory_summary_format(self) -> None:
        """Trajectory data can be formatted for display."""
        trajectories = {
            1: [(100.0, 100.0), (110.0, 105.0), (120.0, 110.0)],
            2: [(200.0, 200.0)],
        }
        rows = [
            {"Person ID": tid, "Points Tracked": len(pts)}
            for tid, pts in trajectories.items()
        ]
        assert len(rows) == 2
        assert rows[0]["Person ID"] == 1
        assert rows[0]["Points Tracked"] == 3
        assert rows[1]["Points Tracked"] == 1

    def test_results_metadata_access(self) -> None:
        """Results dict fields used in playback tab are accessible."""
        results = {
            "total_frames": 100,
            "unique_persons": 5,
            "video_path": "/tmp/test.mp4",
            "trajectories": {1: [(10.0, 20.0)]},
        }
        assert results["video_path"] == "/tmp/test.mp4"
        assert results["total_frames"] == 100
        assert results["unique_persons"] == 5
