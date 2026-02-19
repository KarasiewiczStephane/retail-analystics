"""Tests for the Click CLI module."""

import json
from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from src.cli import cli, main
from src.utils.database import Database


def _create_test_video(path: Path, frames: int = 5) -> None:
    """Create a minimal test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    for _ in range(frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestCliGroup:
    """Tests for the CLI group and basic options."""

    def test_cli_help(self) -> None:
        """CLI shows help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Retail Analytics CLI" in result.output

    def test_cli_version(self) -> None:
        """CLI shows version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_main_is_callable(self) -> None:
        """main entry point is callable."""
        assert callable(main)


class TestProcessCommand:
    """Tests for the process command."""

    def test_process_help(self) -> None:
        """Process command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--confidence" in result.output

    def test_process_missing_input(self) -> None:
        """Process command fails without input."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process"])
        assert result.exit_code != 0

    def test_process_video(self, tmp_path: Path) -> None:
        """Process command runs on a test video."""
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "output"
        _create_test_video(video_path, frames=3)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "process",
                "-i",
                str(video_path),
                "-o",
                str(output_dir),
                "--no-blur-faces",
            ],
        )
        assert result.exit_code == 0
        assert "Processing:" in result.output
        assert "Unique persons:" in result.output
        assert (output_dir / "results.json").exists()
        assert (output_dir / "detections.db").exists()

    def test_process_with_zones(self, tmp_path: Path) -> None:
        """Process command works with zone configuration."""
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "output"
        zones_path = tmp_path / "zones.yaml"
        _create_test_video(video_path, frames=3)

        zones_path.write_text(
            "zones:\n"
            "  - name: entrance\n"
            "    polygon: [[0, 0], [160, 0], [160, 240], [0, 240]]\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "process",
                "-i",
                str(video_path),
                "-c",
                str(zones_path),
                "-o",
                str(output_dir),
                "--no-blur-faces",
            ],
        )
        assert result.exit_code == 0

    def test_process_results_json_format(self, tmp_path: Path) -> None:
        """Results JSON contains expected fields."""
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "output"
        _create_test_video(video_path, frames=3)

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "process",
                "-i",
                str(video_path),
                "-o",
                str(output_dir),
                "--no-blur-faces",
            ],
        )

        results_path = output_dir / "results.json"
        with open(results_path) as f:
            data = json.load(f)
        assert "total_frames" in data
        assert "unique_persons" in data
        assert "detection_count" in data
        assert "generated_at" in data

    def test_process_verbose(self, tmp_path: Path) -> None:
        """Process command accepts verbose flag."""
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "output"
        _create_test_video(video_path, frames=2)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "process",
                "-i",
                str(video_path),
                "-o",
                str(output_dir),
                "--no-blur-faces",
                "-v",
            ],
        )
        assert result.exit_code == 0


class TestReportCommand:
    """Tests for the report command."""

    def test_report_help(self) -> None:
        """Report command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output
        assert "--format" in result.output

    def test_report_from_results_json(self, tmp_path: Path) -> None:
        """Report command generates from results JSON."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.json"
        results_data = {
            "total_frames": 100,
            "unique_persons": 5,
            "detection_count": 50,
        }
        with open(results_path, "w") as f:
            json.dump(results_data, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["report", "-r", str(results_path), "-o", str(output_path)],
        )
        assert result.exit_code == 0
        assert "Report saved to" in result.output
        assert output_path.exists()

        with open(output_path) as f:
            report = json.load(f)
        assert report["total_frames"] == 100
        assert "report_generated_at" in report

    def test_report_csv_format(self, tmp_path: Path) -> None:
        """Report command generates CSV output."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.csv"
        with open(results_path, "w") as f:
            json.dump({"total_frames": 100, "unique_persons": 5}, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "report",
                "-r",
                str(results_path),
                "-o",
                str(output_path),
                "-f",
                "csv",
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "total_frames" in content

    def test_report_from_database(self, tmp_path: Path) -> None:
        """Report command generates from database."""
        db_path = tmp_path / "test.db"
        Database(str(db_path))
        output_path = tmp_path / "report.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["report", "-db", str(db_path), "-o", str(output_path)],
        )
        assert result.exit_code == 0

    def test_report_no_source_fails(self) -> None:
        """Report command fails without results or database."""
        runner = CliRunner()
        result = runner.invoke(cli, ["report"])
        assert result.exit_code != 0


class TestHeatmapCommand:
    """Tests for the heatmap command."""

    def test_heatmap_help(self) -> None:
        """Heatmap command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["heatmap", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--sigma" in result.output

    def test_heatmap_static(self, tmp_path: Path) -> None:
        """Heatmap command generates static PNG."""
        video_path = tmp_path / "test.mp4"
        output_path = tmp_path / "heatmap.png"
        _create_test_video(video_path, frames=3)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "heatmap",
                "-i",
                str(video_path),
                "-o",
                str(output_path),
                "--sigma",
                "10.0",
            ],
        )
        assert result.exit_code == 0
        assert "Heatmap saved to" in result.output
        assert output_path.exists()

    def test_heatmap_missing_input(self) -> None:
        """Heatmap command fails without input."""
        runner = CliRunner()
        result = runner.invoke(cli, ["heatmap", "-o", "out.png"])
        assert result.exit_code != 0

    def test_heatmap_custom_sigma(self, tmp_path: Path) -> None:
        """Heatmap command accepts custom sigma."""
        video_path = tmp_path / "test.mp4"
        output_path = tmp_path / "heatmap.png"
        _create_test_video(video_path, frames=2)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "heatmap",
                "-i",
                str(video_path),
                "-o",
                str(output_path),
                "--sigma",
                "5.0",
            ],
        )
        assert result.exit_code == 0
