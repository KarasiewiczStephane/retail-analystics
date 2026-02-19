"""Click CLI for batch video processing, report generation, and heatmap export.

Provides three commands:
- ``process``: Run the full detection + tracking pipeline on a video.
- ``report``: Generate a JSON analytics report from a results database.
- ``heatmap``: Export a static PNG or animated GIF heatmap.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import cv2
import yaml

from src.analytics.heatmap import HeatmapGenerator
from src.detection.privacy import FaceBlurrer
from src.detection.processor import VideoProcessor
from src.detection.tracker import PersonTracker
from src.utils.database import Database
from src.utils.logger import setup_logger
from src.utils.video_utils import get_video_metadata

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """Retail Analytics CLI - Customer behavior analysis using computer vision."""


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input video file",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="Zones configuration YAML",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--confidence",
    default=0.5,
    type=float,
    help="Detection confidence threshold",
)
@click.option(
    "--blur-faces/--no-blur-faces",
    default=True,
    help="Apply face blurring",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def process(
    input_path: str,
    config_path: Optional[str],
    output_dir: Optional[str],
    confidence: float,
    blur_faces: bool,
    verbose: bool,
) -> None:
    """Process a video file for retail analytics.

    Example:
        retail-analytics process -i video.mp4 -c zones.yaml -o output/
    """
    setup_logger("src", level="DEBUG" if verbose else "INFO")

    out = Path(output_dir) if output_dir else Path(input_path).parent / "output"
    out.mkdir(parents=True, exist_ok=True)

    db_path = out / "detections.db"
    db = Database(str(db_path))
    tracker = PersonTracker(confidence_threshold=confidence)
    processor = VideoProcessor(db, tracker)
    metadata = get_video_metadata(input_path)

    zone_analyzer = None
    zone_names: list[str] = []
    if config_path:
        with open(config_path) as f:
            zones_data = yaml.safe_load(f)
        if zones_data and "zones" in zones_data:
            from src.analytics.zone_analyzer import ZoneAnalyzer

            zone_analyzer = ZoneAnalyzer.from_config(zones_data["zones"])
            zone_names = list(zone_analyzer.zones.keys())

    click.echo(f"Processing: {input_path}")
    click.echo(
        f"  Resolution: {metadata['width']}x{metadata['height']} | "
        f"FPS: {metadata['fps']:.1f} | Frames: {metadata['total_frames']}"
    )

    with click.progressbar(
        length=metadata["total_frames"], label="Processing video"
    ) as bar:
        last_pos = 0

        def progress_callback(current: int, total: int) -> None:
            nonlocal last_pos
            delta = current - last_pos
            if delta > 0:
                bar.update(delta)
                last_pos = current

        results = processor.process_video(
            input_path,
            progress_callback=progress_callback,
        )

    if zone_analyzer and zone_names:
        from src.analytics.dwell_time import DwellTimeTracker, ZoneTransitionMatrix
        from src.analytics.traffic_counter import TrafficCounter

        dwell_tracker = DwellTimeTracker(fps=metadata["fps"])
        transition_matrix = ZoneTransitionMatrix(zone_names)
        traffic_counter = TrafficCounter(fps=metadata["fps"])

        for tid, trajectory in results["trajectories"].items():
            for i, (cx, cy) in enumerate(trajectory):
                events = zone_analyzer.update(int(tid), (cx, cy), i)
                for zn, evt in events.items():
                    if evt == "enter":
                        dwell_tracker.start_dwell(int(tid), zn, i)
                        transition_matrix.record_zone_entry(int(tid), zn)
                        traffic_counter.record_entry(i)
                    elif evt == "exit":
                        dwell_tracker.end_dwell(int(tid), zn, i)
                        transition_matrix.record_zone_exit(int(tid), zn)
        dwell_tracker.finalize_active_dwells(metadata["total_frames"])

        results["zone_metrics"] = {
            name: {
                "unique_visitors": m.unique_visitors,
                "total_entries": m.total_entries,
                "avg_dwell_time": dwell_tracker.get_zone_average_dwell_time(name),
            }
            for name, m in zone_analyzer.get_all_metrics().items()
        }
        results["hourly_distribution"] = traffic_counter.get_hourly_distribution()

    if blur_faces:
        click.echo("Applying face blurring...")
        blurrer = FaceBlurrer()
        blurred_path = out / f"{Path(input_path).stem}_blurred.mp4"
        blur_result = blurrer.process_video(input_path, str(blurred_path))
        results["blurred_video"] = str(blurred_path)
        click.echo(f"  Faces blurred: {blur_result['faces_blurred']}")

    results_path = out / "results.json"
    serializable = {
        "video_id": results.get("video_id", Path(input_path).stem),
        "total_frames": results["total_frames"],
        "unique_persons": results["unique_persons"],
        "detection_count": results["detection_count"],
        "generated_at": datetime.now().isoformat(),
    }
    if "zone_metrics" in results:
        serializable["zone_metrics"] = results["zone_metrics"]
    if "hourly_distribution" in results:
        serializable["hourly_distribution"] = results["hourly_distribution"]

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    click.echo(f"\nResults saved to: {out}")
    click.echo(f"  Unique persons: {results['unique_persons']}")
    click.echo(f"  Total detections: {results['detection_count']}")
    click.echo(f"  Database: {db_path}")


@cli.command()
@click.option(
    "--database",
    "-db",
    type=click.Path(exists=True),
    help="Database path",
)
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True),
    help="Results JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Output format",
)
def report(
    database: Optional[str],
    results: Optional[str],
    output: Optional[str],
    fmt: str,
) -> None:
    """Generate analytics report from processing results.

    Example:
        retail-analytics report -r output/results.json -o report.json
    """
    report_data: dict = {}

    if results:
        with open(results) as f:
            report_data = json.load(f)
    elif database:
        db = Database(database)
        detections = db.get_detections_by_video("default")
        person_ids = {d["person_id"] for d in detections}
        report_data["unique_persons"] = len(person_ids)
        report_data["total_detections"] = len(detections)
    else:
        click.echo("Provide --results or --database", err=True)
        raise SystemExit(1)

    report_data["report_generated_at"] = datetime.now().isoformat()

    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"report.{fmt}")

    if fmt == "json":
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
    elif fmt == "csv":
        import pandas as pd

        flat = {k: v for k, v in report_data.items() if not isinstance(v, (dict, list))}
        df = pd.DataFrame([flat])
        df.to_csv(output_path, index=False)

    click.echo(f"Report saved to: {output_path}")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input video file",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output heatmap path (PNG/GIF)",
)
@click.option(
    "--layout",
    "-l",
    type=click.Path(exists=True),
    help="Store layout image for overlay",
)
@click.option(
    "--animated/--static",
    default=False,
    help="Generate animated GIF",
)
@click.option(
    "--sigma",
    default=20.0,
    type=float,
    help="Gaussian smoothing sigma",
)
@click.option(
    "--confidence",
    default=0.5,
    type=float,
    help="Detection confidence threshold",
)
def heatmap(
    input_path: str,
    output_path: str,
    layout: Optional[str],
    animated: bool,
    sigma: float,
    confidence: float,
) -> None:
    """Generate heatmap from a video file.

    Example:
        retail-analytics heatmap -i video.mp4 -o heatmap.png
        retail-analytics heatmap -i video.mp4 -o heatmap.gif --animated
    """
    click.echo(f"Generating heatmap from: {input_path}")

    db = Database(":memory:")
    tracker = PersonTracker(confidence_threshold=confidence)
    processor = VideoProcessor(db, tracker)
    metadata = get_video_metadata(input_path)

    with click.progressbar(
        length=metadata["total_frames"], label="Processing video"
    ) as bar:
        last_pos = 0

        def progress_callback(current: int, total: int) -> None:
            nonlocal last_pos
            delta = current - last_pos
            if delta > 0:
                bar.update(delta)
                last_pos = current

        results = processor.process_video(
            input_path,
            progress_callback=progress_callback,
        )

    trajectories = results.get("trajectories", {})
    width = metadata["width"]
    height = metadata["height"]

    background = None
    if layout:
        background = cv2.imread(layout)

    generator = HeatmapGenerator.from_trajectories(
        trajectories, width, height, sigma=sigma
    )

    if animated:
        window_size = max(1, len(trajectories) // 10) if trajectories else 1
        points_added = 0
        for points in trajectories.values():
            for x, y in points:
                generator.current_window[int(y), int(x)] += 1
                points_added += 1
                if points_added % window_size == 0:
                    generator.end_time_window()
        if points_added % window_size != 0:
            generator.end_time_window()
        generator.export_animated_gif(output_path, background=background)
    else:
        generator.export_png(output_path, background=background)

    click.echo(f"Heatmap saved to: {output_path}")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
