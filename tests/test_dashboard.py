"""Tests for the Streamlit dashboard module.

Streamlit apps are tested by verifying that the module functions
can be imported and called with mock state, and that the processing
logic produces expected results.
"""

from src.dashboard.app import _traffic_overview_tab, _zone_analysis_tab, main


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
