# Retail Analytics

[![CI](https://github.com/KarasiewiczStephane/retail-analystics/workflows/CI/badge.svg)](https://github.com/KarasiewiczStephane/retail-analystics/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

YOLOv8-based system for customer behavior analysis in retail environments. Tracks foot traffic, calculates dwell time per zone, generates heatmaps, and includes privacy-preserving face blurring.

## Features

- **Person Detection & Tracking** - YOLOv8 + ByteTrack for consistent person identification across frames
- **Zone Analytics** - Configurable polygon zones with foot traffic counting, dwell time, and transition analysis
- **Heatmap Generation** - Gaussian-smoothed heatmaps with animated time progression (PNG/GIF)
- **Privacy Pipeline** - Automatic face blurring (YOLO-face with Haar cascade fallback) on all exported videos
- **Interactive Dashboard** - Streamlit web interface with upload, processing, traffic overview, zone analysis, and heatmap visualization
- **CLI Interface** - Batch processing, report generation, and heatmap export via Click
- **Docker Support** - Multi-stage build with dashboard and CLI services

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Video Input │────>│ YOLOv8       │────>│ ByteTrack   │
└─────────────┘     │ Detection    │     │ Tracking    │
                    └──────────────┘     └──────┬──────┘
                                                │
          ┌─────────────────┬───────────────────┼────────────────┐
          v                 v                   v                v
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐
   │ Zone         │  │ Dwell Time   │  │ Heatmap      │  │ Face      │
   │ Analyzer     │  │ Tracker      │  │ Generator    │  │ Blurrer   │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘
          │                 │                  │                │
          └─────────────────┴──────────────────┴────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  v                                   v
           ┌──────────────┐                    ┌──────────────┐
           │ Streamlit    │                    │ Click CLI    │
           │ Dashboard    │                    │ Interface    │
           └──────────────┘                    └──────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
git clone git@github.com:KarasiewiczStephane/retail-analystics.git
cd retail-analystics
make install
```

### 2. Launch the dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

From the dashboard you can upload a video, optionally attach a zones YAML config, adjust detection confidence and blur settings in the sidebar, then click **Process Video**. Results appear across the Traffic Overview, Zone Analysis, Heatmap View, and Video Playback tabs.

No separate data preparation step is required -- videos are processed on upload.

### 3. CLI Usage (alternative)

```bash
# Process a video with zone analysis
make run ARGS="process -i video.mp4 -c configs/zones_example.yaml -o output/"

# Or call the module directly
python -m src.cli process -i video.mp4 -c configs/zones_example.yaml -o output/

# Generate a report from results
python -m src.cli report -r output/results.json -o report.json

# Create a heatmap (static PNG or animated GIF)
python -m src.cli heatmap -i video.mp4 -o heatmap.png --sigma 15.0
python -m src.cli heatmap -i video.mp4 -o heatmap.gif --animated
```

### Docker

```bash
# Build and run the dashboard
docker compose up -d
# Dashboard available at http://localhost:8501

# Or run CLI commands
make docker-cli ARGS="process -i /app/input/video.mp4 -o /app/output/"
```

## Configuration

### Zone Definition

Create a YAML file defining store zones as polygons (see `configs/zones_example.yaml`):

```yaml
zones:
  - name: entrance
    polygon: [[100, 100], [300, 100], [300, 400], [100, 400]]
    color: [0, 255, 0]
  - name: checkout
    polygon: [[500, 200], [700, 200], [700, 500], [500, 500]]
    color: [255, 0, 0]
  - name: electronics
    polygon: [[350, 50], [600, 50], [600, 350], [350, 350]]
    color: [0, 0, 255]
```

### Application Config

Central configuration in `configs/config.yaml` covers detection thresholds, tracking parameters, privacy settings, database paths, heatmap options, and dashboard defaults.

## Project Structure

```
retail-analystics/
├── src/
│   ├── detection/        # YOLOv8 detector, ByteTrack tracker, video processor, privacy pipeline
│   ├── analytics/        # Zone analyzer, dwell time, traffic counter, heatmap generator
│   ├── dashboard/        # Streamlit web application
│   ├── utils/            # Config loader, database, logger, video I/O
│   └── cli.py            # Click CLI entry point
├── tests/                # Unit tests (216+ tests, 80%+ coverage)
├── configs/              # YAML configuration files
├── .github/workflows/    # CI pipeline (lint, test, Docker build)
├── Dockerfile            # Multi-stage production build
├── docker-compose.yml    # Dashboard + CLI services
├── Makefile              # Build, test, lint, Docker commands
├── requirements.txt      # Python dependencies
└── pytest.ini            # Test configuration
```

## Development

```bash
# Run tests
make test

# Lint and format
make lint

# Clean caches
make clean
```

## Sample Output

After processing a video with zone configuration:

- **Traffic Overview**: Total visitors, detection count, hourly distribution chart
- **Zone Analysis**: Per-zone visitor count, entry count, average dwell time, transition heatmap
- **Heatmap**: Gaussian-smoothed spatial density map of person trajectories
- **Privacy**: Face-blurred output video

## License

MIT
