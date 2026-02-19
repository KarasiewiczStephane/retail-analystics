"""Tests for documentation and sample data completeness."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


class TestReadme:
    """Tests for README.md content."""

    def test_readme_exists(self) -> None:
        """README.md exists in project root."""
        assert (ROOT / "README.md").exists()

    def test_readme_has_title(self) -> None:
        """README has a project title."""
        content = (ROOT / "README.md").read_text()
        assert "# Retail Analytics" in content

    def test_readme_has_features(self) -> None:
        """README lists project features."""
        content = (ROOT / "README.md").read_text()
        assert "## Features" in content
        assert "Person Detection" in content
        assert "Zone Analytics" in content
        assert "Heatmap" in content
        assert "Privacy" in content

    def test_readme_has_architecture(self) -> None:
        """README includes architecture diagram."""
        content = (ROOT / "README.md").read_text()
        assert "## Architecture" in content
        assert "YOLOv8" in content
        assert "ByteTrack" in content

    def test_readme_has_quick_start(self) -> None:
        """README has quick start instructions."""
        content = (ROOT / "README.md").read_text()
        assert "## Quick Start" in content
        assert "pip install" in content

    def test_readme_has_cli_usage(self) -> None:
        """README documents CLI usage."""
        content = (ROOT / "README.md").read_text()
        assert "CLI Usage" in content
        assert "process" in content
        assert "report" in content
        assert "heatmap" in content

    def test_readme_has_docker_section(self) -> None:
        """README documents Docker usage."""
        content = (ROOT / "README.md").read_text()
        assert "Docker" in content
        assert "docker compose" in content

    def test_readme_has_project_structure(self) -> None:
        """README shows project structure."""
        content = (ROOT / "README.md").read_text()
        assert "## Project Structure" in content
        assert "src/" in content
        assert "tests/" in content

    def test_readme_has_configuration(self) -> None:
        """README documents configuration options."""
        content = (ROOT / "README.md").read_text()
        assert "## Configuration" in content
        assert "zones" in content

    def test_readme_has_ci_badge(self) -> None:
        """README has CI status badge."""
        content = (ROOT / "README.md").read_text()
        assert "badge.svg" in content

    def test_readme_has_license(self) -> None:
        """README mentions license."""
        content = (ROOT / "README.md").read_text()
        assert "MIT" in content


class TestConfigFiles:
    """Tests for configuration files."""

    def test_config_yaml_exists(self) -> None:
        """Main config.yaml exists."""
        assert (ROOT / "configs" / "config.yaml").exists()

    def test_config_yaml_valid(self) -> None:
        """config.yaml is valid YAML."""
        path = ROOT / "configs" / "config.yaml"
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict)

    def test_zones_example_exists(self) -> None:
        """zones_example.yaml exists."""
        assert (ROOT / "configs" / "zones_example.yaml").exists()

    def test_zones_example_valid(self) -> None:
        """zones_example.yaml has valid zone definitions."""
        path = ROOT / "configs" / "zones_example.yaml"
        data = yaml.safe_load(path.read_text())
        assert "zones" in data
        assert len(data["zones"]) > 0
        for zone in data["zones"]:
            assert "name" in zone
            assert "polygon" in zone


class TestSampleDataGenerator:
    """Tests for the sample video generator."""

    def test_generator_script_exists(self) -> None:
        """Sample video generator script exists."""
        assert (ROOT / "scripts" / "generate_sample_video.py").exists()

    def test_generator_is_importable(self) -> None:
        """Generator function can be imported."""
        from scripts.generate_sample_video import generate_sample_video

        assert callable(generate_sample_video)

    def test_generator_creates_video(self, tmp_path: Path) -> None:
        """Generator creates a valid video file."""
        from scripts.generate_sample_video import generate_sample_video

        output = tmp_path / "test.mp4"
        generate_sample_video(str(output), duration_seconds=1.0)
        assert output.exists()
        assert output.stat().st_size > 0


class TestProjectFiles:
    """Tests for essential project files."""

    def test_requirements_txt_exists(self) -> None:
        """requirements.txt exists."""
        assert (ROOT / "requirements.txt").exists()

    def test_requirements_has_core_deps(self) -> None:
        """requirements.txt lists core dependencies."""
        content = (ROOT / "requirements.txt").read_text()
        assert "ultralytics" in content
        assert "streamlit" in content
        assert "click" in content
        assert "opencv" in content.lower() or "cv2" in content

    def test_gitignore_exists(self) -> None:
        """gitignore file exists."""
        assert (ROOT / ".gitignore").exists()

    def test_env_example_exists(self) -> None:
        """env.example file exists."""
        assert (ROOT / ".env.example").exists()

    def test_makefile_exists(self) -> None:
        """Makefile exists."""
        assert (ROOT / "Makefile").exists()

    def test_precommit_config_exists(self) -> None:
        """Pre-commit config exists."""
        assert (ROOT / ".pre-commit-config.yaml").exists()
