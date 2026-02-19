"""Tests for the heatmap generation pipeline."""

from pathlib import Path

import cv2
import numpy as np

from src.analytics.heatmap import HeatmapGenerator


class TestHeatmapGenerator:
    """Tests for the HeatmapGenerator class."""

    def test_add_point_accumulates(self) -> None:
        """Adding a point increases the accumulator value."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        assert gen.accumulator[50, 50] == 1.0

    def test_add_point_out_of_bounds(self) -> None:
        """Out-of-bounds points are ignored."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(200, 200)
        assert gen.accumulator.sum() == 0.0

    def test_add_point_weight(self) -> None:
        """Custom weight is applied to the accumulator."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50, weight=3.0)
        assert gen.accumulator[50, 50] == 3.0

    def test_add_points_batch(self) -> None:
        """Batch add accumulates all points."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_points_batch([(10, 10), (20, 20), (30, 30)])
        assert gen.accumulator[10, 10] == 1.0
        assert gen.accumulator[20, 20] == 1.0
        assert gen.accumulator[30, 30] == 1.0

    def test_generate_heatmap_normalized(self) -> None:
        """Normalized heatmap has values in [0, 1]."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50, weight=10.0)
        heatmap = gen.generate_heatmap(normalize=True)
        assert heatmap.max() <= 1.0
        assert heatmap.min() >= 0.0

    def test_generate_heatmap_unnormalized(self) -> None:
        """Unnormalized heatmap preserves raw smoothed values."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50, weight=10.0)
        heatmap = gen.generate_heatmap(normalize=False)
        assert heatmap.max() > 0

    def test_generate_heatmap_empty(self) -> None:
        """Empty accumulator produces an all-zero heatmap."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        heatmap = gen.generate_heatmap()
        assert heatmap.max() == 0.0

    def test_generate_colored_heatmap_shape(self) -> None:
        """Colored heatmap has correct BGR shape."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        colored = gen.generate_colored_heatmap()
        assert colored.shape == (100, 100, 3)
        assert colored.dtype == np.uint8

    def test_overlay_on_image(self) -> None:
        """Overlay produces a blended image matching background dimensions."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        result = gen.overlay_on_image(bg)
        assert result.shape == (100, 100, 3)

    def test_overlay_different_sizes(self) -> None:
        """Overlay handles background with different dimensions."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        result = gen.overlay_on_image(bg)
        assert result.shape == (200, 200, 3)

    def test_export_png(self, tmp_path: Path) -> None:
        """Export produces a valid PNG file."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        output = tmp_path / "heatmap.png"
        gen.export_png(str(output))
        assert output.exists()
        img = cv2.imread(str(output))
        assert img is not None
        assert img.shape == (100, 100, 3)

    def test_export_png_with_background(self, tmp_path: Path) -> None:
        """Export with background produces an overlay image."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50)
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 128
        output = tmp_path / "heatmap_overlay.png"
        gen.export_png(str(output), background=bg)
        assert output.exists()

    def test_time_windows(self) -> None:
        """Time windows capture accumulator snapshots."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(10, 10)
        gen.end_time_window()
        gen.add_point(20, 20)
        gen.end_time_window()
        assert len(gen.time_windows) == 2

    def test_export_animated_gif(self, tmp_path: Path) -> None:
        """Export produces a valid animated GIF."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(10, 10)
        gen.end_time_window()
        gen.add_point(50, 50)
        gen.end_time_window()
        output = tmp_path / "heatmap.gif"
        gen.export_animated_gif(str(output), fps=2)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_export_animated_gif_empty(self, tmp_path: Path) -> None:
        """Empty time windows do not create a GIF."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        output = tmp_path / "empty.gif"
        gen.export_animated_gif(str(output))
        assert not output.exists()

    def test_from_trajectories(self) -> None:
        """Factory method creates a generator with accumulated trajectories."""
        trajectories = {
            1: [(10.0, 10.0), (20.0, 20.0)],
            2: [(50.0, 50.0)],
        }
        gen = HeatmapGenerator.from_trajectories(trajectories, 100, 100, sigma=5.0)
        assert gen.accumulator[10, 10] == 1.0
        assert gen.accumulator[50, 50] == 1.0

    def test_gaussian_smoothing_spreads_values(self) -> None:
        """Gaussian smoothing spreads point values to neighbors."""
        gen = HeatmapGenerator(100, 100, sigma=5.0)
        gen.add_point(50, 50, weight=100.0)
        heatmap = gen.generate_heatmap(normalize=False)
        assert heatmap[50, 50] > heatmap[50, 55]
        assert heatmap[50, 55] > 0
