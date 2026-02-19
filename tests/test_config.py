"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    AppConfig,
    DetectionConfig,
    ZoneDefinition,
    load_config,
    load_zones,
)


@pytest.fixture
def valid_config_file(tmp_path: Path) -> Path:
    """Create a temporary valid config YAML file."""
    config = {
        "detection": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.6,
            "person_class_id": 0,
        },
        "tracking": {"tracker_type": "bytetrack.yaml", "persist": True},
        "privacy": {
            "face_model_path": "yolov8n-face.pt",
            "blur_intensity": 31,
        },
        "database": {"path": "data/test.db"},
        "heatmap": {"sigma": 15.0},
        "logging": {"level": "DEBUG"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def valid_zones_file(tmp_path: Path) -> Path:
    """Create a temporary valid zones YAML file."""
    zones = {
        "zones": [
            {
                "name": "entrance",
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "color": [0, 255, 0],
            },
            {
                "name": "checkout",
                "polygon": [[200, 200], [400, 200], [400, 400], [200, 400]],
            },
        ]
    }
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(yaml.dump(zones))
    return zones_path


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, valid_config_file: Path) -> None:
        """Valid YAML config loads correctly into AppConfig."""
        config = load_config(str(valid_config_file))
        assert isinstance(config, AppConfig)
        assert config.detection.confidence_threshold == 0.6
        assert config.privacy.blur_intensity == 31
        assert config.database.path == "data/test.db"
        assert config.heatmap.sigma == 15.0

    def test_load_config_defaults(self, tmp_path: Path) -> None:
        """Empty YAML file produces default AppConfig values."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        config = load_config(str(config_path))
        assert config.detection.confidence_threshold == 0.5
        assert config.detection.model_path == "yolov8n.pt"

    def test_load_config_missing_file(self) -> None:
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_partial(self, tmp_path: Path) -> None:
        """Config with only some sections uses defaults for the rest."""
        config_data = {"detection": {"confidence_threshold": 0.8}}
        config_path = tmp_path / "partial.yaml"
        config_path.write_text(yaml.dump(config_data))
        config = load_config(str(config_path))
        assert config.detection.confidence_threshold == 0.8
        assert config.database.path == "data/detections.db"

    def test_default_detection_config(self) -> None:
        """DetectionConfig defaults are correct."""
        dc = DetectionConfig()
        assert dc.model_path == "yolov8n.pt"
        assert dc.confidence_threshold == 0.5
        assert dc.person_class_id == 0


class TestLoadZones:
    """Tests for the load_zones function."""

    def test_load_valid_zones(self, valid_zones_file: Path) -> None:
        """Valid zones file loads all zone definitions."""
        zones = load_zones(str(valid_zones_file))
        assert len(zones) == 2
        assert zones[0].name == "entrance"
        assert zones[0].polygon == [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert zones[0].color == [0, 255, 0]

    def test_zone_default_color(self, valid_zones_file: Path) -> None:
        """Zones without explicit color get the default green."""
        zones = load_zones(str(valid_zones_file))
        assert zones[1].color == [0, 255, 0]

    def test_load_zones_missing_file(self) -> None:
        """Missing zones file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_zones("/nonexistent/zones.yaml")

    def test_load_zones_missing_key(self, tmp_path: Path) -> None:
        """Zones file without 'zones' key raises ValueError."""
        zones_path = tmp_path / "bad.yaml"
        zones_path.write_text(yaml.dump({"other": "data"}))
        with pytest.raises(ValueError, match="missing 'zones' key"):
            load_zones(str(zones_path))

    def test_load_zones_missing_name(self, tmp_path: Path) -> None:
        """Zone definition missing 'name' raises ValueError."""
        zones_data = {"zones": [{"polygon": [[0, 0], [1, 0], [1, 1]]}]}
        zones_path = tmp_path / "noname.yaml"
        zones_path.write_text(yaml.dump(zones_data))
        with pytest.raises(ValueError, match="missing 'name' or 'polygon'"):
            load_zones(str(zones_path))

    def test_load_zones_missing_polygon(self, tmp_path: Path) -> None:
        """Zone definition missing 'polygon' raises ValueError."""
        zones_data = {"zones": [{"name": "test"}]}
        zones_path = tmp_path / "nopolygon.yaml"
        zones_path.write_text(yaml.dump(zones_data))
        with pytest.raises(ValueError, match="missing 'name' or 'polygon'"):
            load_zones(str(zones_path))

    def test_zone_definition_dataclass(self) -> None:
        """ZoneDefinition dataclass stores correct values."""
        zone = ZoneDefinition(name="test", polygon=[[0, 0], [1, 1]], color=[255, 0, 0])
        assert zone.name == "test"
        assert zone.polygon == [[0, 0], [1, 1]]
        assert zone.color == [255, 0, 0]
