"""Configuration management for retail analytics.

Loads and validates YAML configuration files for detection settings,
zone definitions, and application parameters.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for YOLOv8 person detection."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    person_class_id: int = 0


@dataclass
class TrackingConfig:
    """Configuration for ByteTrack tracking."""

    tracker_type: str = "bytetrack.yaml"
    persist: bool = True


@dataclass
class PrivacyConfig:
    """Configuration for face blurring pipeline."""

    face_model_path: str = "yolov8n-face.pt"
    face_confidence_threshold: float = 0.3
    blur_intensity: int = 51
    enable_face_blurring: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for SQLite database."""

    path: str = "data/detections.db"


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""

    sigma: float = 20.0
    colormap: str = "jet"
    alpha: float = 0.6


@dataclass
class VideoConfig:
    """Configuration for video input handling."""

    supported_formats: list[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov"]
    )


@dataclass
class LoggingConfig:
    """Configuration for application logging."""

    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class DashboardConfig:
    """Configuration for Streamlit dashboard."""

    host: str = "0.0.0.0"
    port: int = 8501


@dataclass
class ZoneDefinition:
    """A single zone defined by a polygon and display color."""

    name: str
    polygon: list[list[int]]
    color: list[int] = field(default_factory=lambda: [0, 255, 0])


@dataclass
class AppConfig:
    """Top-level application configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def load_config(config_path: str) -> AppConfig:
    """Load application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Populated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        logger.warning("Empty config file, using defaults")
        return AppConfig()

    config = AppConfig(
        detection=DetectionConfig(**raw.get("detection", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        privacy=PrivacyConfig(**raw.get("privacy", {})),
        database=DatabaseConfig(**raw.get("database", {})),
        heatmap=HeatmapConfig(**raw.get("heatmap", {})),
        video=VideoConfig(**raw.get("video", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        dashboard=DashboardConfig(**raw.get("dashboard", {})),
    )

    logger.info("Configuration loaded from %s", config_path)
    return config


def load_zones(zones_path: str) -> list[ZoneDefinition]:
    """Load zone definitions from a YAML file.

    Args:
        zones_path: Path to the zones YAML configuration file.

    Returns:
        List of ZoneDefinition instances.

    Raises:
        FileNotFoundError: If the zones file does not exist.
        ValueError: If the zones file is missing required fields.
    """
    path = Path(zones_path)
    if not path.exists():
        raise FileNotFoundError(f"Zones file not found: {zones_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None or "zones" not in raw:
        raise ValueError(f"Invalid zones file: missing 'zones' key in {zones_path}")

    zones = []
    for zone_data in raw["zones"]:
        if "name" not in zone_data or "polygon" not in zone_data:
            raise ValueError(
                f"Zone definition missing 'name' or 'polygon': {zone_data}"
            )
        zones.append(
            ZoneDefinition(
                name=zone_data["name"],
                polygon=zone_data["polygon"],
                color=zone_data.get("color", [0, 255, 0]),
            )
        )

    logger.info("Loaded %d zones from %s", len(zones), zones_path)
    return zones
