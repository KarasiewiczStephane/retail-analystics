"""Heatmap generation from detection centroids.

Accumulates person positions over time, applies Gaussian smoothing,
and produces colored heatmap images (PNG) and animated GIFs.
"""

import logging
from typing import Optional

import cv2
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Builds and exports heatmaps from person detection centroids.

    Accumulates detection positions into a 2D grid and applies
    Gaussian smoothing for visualization.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        sigma: Standard deviation for the Gaussian smoothing kernel.
    """

    def __init__(self, width: int, height: int, sigma: float = 20.0) -> None:
        self.width = width
        self.height = height
        self.sigma = sigma
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self.time_windows: list[np.ndarray] = []
        self.current_window = np.zeros((height, width), dtype=np.float32)

    def add_point(self, x: float, y: float, weight: float = 1.0) -> None:
        """Add a detection centroid to the accumulator.

        Points outside the frame boundaries are silently ignored.

        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
            weight: Contribution weight for this point.
        """
        ix, iy = int(x), int(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            self.accumulator[iy, ix] += weight
            self.current_window[iy, ix] += weight

    def add_points_batch(
        self, points: list[tuple[float, float]], weight: float = 1.0
    ) -> None:
        """Add multiple centroids to the accumulator.

        Args:
            points: List of ``(x, y)`` coordinates.
            weight: Contribution weight for each point.
        """
        for x, y in points:
            self.add_point(x, y, weight)

    def end_time_window(self) -> None:
        """Save the current time window and start a new one.

        Used for building animated heatmap progressions.
        """
        self.time_windows.append(self.current_window.copy())
        self.current_window = np.zeros((self.height, self.width), dtype=np.float32)

    def generate_heatmap(self, normalize: bool = True) -> np.ndarray:
        """Generate a Gaussian-smoothed heatmap.

        Args:
            normalize: If True, scale values to the ``[0, 1]`` range.

        Returns:
            2D numpy array of smoothed heatmap values.
        """
        smoothed = gaussian_filter(self.accumulator, sigma=self.sigma)
        if normalize and smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        return smoothed

    def generate_colored_heatmap(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Generate a BGR colored heatmap image.

        Args:
            colormap: OpenCV colormap constant (default ``cv2.COLORMAP_JET``).

        Returns:
            BGR image as a ``(H, W, 3)`` uint8 numpy array.
        """
        heatmap = self.generate_heatmap()
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        return cv2.applyColorMap(heatmap_uint8, colormap)

    def overlay_on_image(
        self,
        background: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay the heatmap on a background image.

        Args:
            background: BGR background image.
            alpha: Blending weight for the heatmap layer.
            colormap: OpenCV colormap constant.

        Returns:
            Blended BGR image.
        """
        heatmap_colored = self.generate_colored_heatmap(colormap)
        if heatmap_colored.shape[:2] != background.shape[:2]:
            heatmap_colored = cv2.resize(
                heatmap_colored, (background.shape[1], background.shape[0])
            )
        return cv2.addWeighted(background, 1 - alpha, heatmap_colored, alpha, 0)

    def export_png(
        self,
        output_path: str,
        background: Optional[np.ndarray] = None,
    ) -> None:
        """Export the heatmap as a PNG image.

        Args:
            output_path: File path for the output PNG.
            background: Optional background image for overlay.
        """
        if background is not None:
            img = self.overlay_on_image(background)
        else:
            img = self.generate_colored_heatmap()
        cv2.imwrite(output_path, img)
        logger.info("Heatmap exported to %s", output_path)

    def export_animated_gif(
        self,
        output_path: str,
        fps: int = 2,
        background: Optional[np.ndarray] = None,
    ) -> None:
        """Export an animated heatmap progression as a GIF.

        Each frame of the GIF shows the cumulative heatmap up to that
        time window.

        Args:
            output_path: File path for the output GIF.
            fps: Frames per second for the animation.
            background: Optional background image for overlay.
        """
        frames: list[np.ndarray] = []
        cumulative = np.zeros((self.height, self.width), dtype=np.float32)

        for window in self.time_windows:
            cumulative += window
            smoothed = gaussian_filter(cumulative, sigma=self.sigma)
            if smoothed.max() > 0:
                smoothed = smoothed / smoothed.max()
            heatmap_uint8 = (smoothed * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            if background is not None:
                bg = background.copy()
                if colored.shape[:2] != bg.shape[:2]:
                    colored = cv2.resize(colored, (bg.shape[1], bg.shape[0]))
                colored = cv2.addWeighted(bg, 0.4, colored, 0.6, 0)

            frames.append(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))

        if frames:
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
            logger.info(
                "Animated heatmap exported to %s (%d frames)", output_path, len(frames)
            )

    @classmethod
    def from_trajectories(
        cls,
        trajectories: dict[int, list[tuple[float, float]]],
        width: int,
        height: int,
        sigma: float = 20.0,
    ) -> "HeatmapGenerator":
        """Create a heatmap from pre-computed person trajectories.

        Args:
            trajectories: Dictionary mapping track IDs to centroid lists.
            width: Frame width in pixels.
            height: Frame height in pixels.
            sigma: Gaussian smoothing sigma.

        Returns:
            HeatmapGenerator with all trajectory points accumulated.
        """
        generator = cls(width, height, sigma)
        for points in trajectories.values():
            generator.add_points_batch(points)
        return generator
