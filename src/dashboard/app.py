"""Streamlit dashboard for retail analytics.

Provides video upload, zone configuration, real-time processing with
progress feedback, traffic overview with visitor counts and hourly
distribution charts, and per-zone analysis with transition heatmaps.
"""

import tempfile

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="Retail Analytics", layout="wide")

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "zone_metrics" not in st.session_state:
        st.session_state.zone_metrics = None
    if "transition_matrix" not in st.session_state:
        st.session_state.transition_matrix = None
    if "zone_names" not in st.session_state:
        st.session_state.zone_names = None

    st.title("Retail Analytics Dashboard")

    with st.sidebar:
        st.header("Configuration")
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
        blur_intensity = st.slider("Blur Intensity", 11, 99, 51, 2)
        zones_file = st.file_uploader(
            "Upload Zones Config (YAML)", type=["yaml", "yml"]
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Upload & Process",
            "Traffic Overview",
            "Zone Analysis",
            "Heatmap View",
            "Video Playback",
        ]
    )

    with tab1:
        _upload_and_process_tab(confidence, blur_intensity, zones_file)
    with tab2:
        _traffic_overview_tab()
    with tab3:
        _zone_analysis_tab()
    with tab4:
        _heatmap_view_tab()
    with tab5:
        _video_playback_tab()


def _upload_and_process_tab(
    confidence: float,
    blur_intensity: int,
    zones_file,
) -> None:
    """Render the video upload and processing tab.

    Args:
        confidence: Detection confidence threshold from the sidebar slider.
        blur_intensity: Face blur kernel size from the sidebar slider.
        zones_file: Uploaded zones YAML file or None.
    """
    st.header("Upload & Process Video")
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name

        st.video(video_path)

        if st.button("Process Video"):
            _run_processing(video_path, confidence, blur_intensity, zones_file)


def _run_processing(
    video_path: str,
    confidence: float,
    blur_intensity: int,
    zones_file,
) -> None:
    """Execute the full processing pipeline on an uploaded video.

    Args:
        video_path: Path to the temporary video file.
        confidence: Detection confidence threshold.
        blur_intensity: Face blur kernel size.
        zones_file: Uploaded zones YAML file or None.
    """
    from src.analytics.dwell_time import DwellTimeTracker, ZoneTransitionMatrix
    from src.analytics.traffic_counter import TrafficCounter
    from src.analytics.zone_analyzer import ZoneAnalyzer
    from src.detection.processor import VideoProcessor
    from src.detection.tracker import PersonTracker
    from src.utils.database import Database
    from src.utils.video_utils import get_video_metadata

    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current: int, total: int) -> None:
            if total > 0:
                progress_bar.progress(current / total)
                status_text.text(f"Processing frame {current}/{total}")

        db = Database(":memory:")
        tracker = PersonTracker(confidence_threshold=confidence)
        processor = VideoProcessor(db, tracker)
        metadata = get_video_metadata(video_path)

        zone_analyzer = None
        dwell_tracker = None
        transition_matrix = None
        zone_names: list[str] = []

        if zones_file is not None:
            zones_data = yaml.safe_load(zones_file)
            if zones_data and "zones" in zones_data:
                zone_analyzer = ZoneAnalyzer.from_config(zones_data["zones"])
                zone_names = list(zone_analyzer.zones.keys())
                dwell_tracker = DwellTimeTracker(fps=metadata["fps"])
                transition_matrix = ZoneTransitionMatrix(zone_names)

        traffic_counter = TrafficCounter(fps=metadata["fps"])
        results = processor.process_video(
            video_path,
            progress_callback=update_progress,
        )

        if zone_analyzer and dwell_tracker and transition_matrix:
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

            st.session_state.zone_metrics = {
                name: {
                    "unique_visitors": m.unique_visitors,
                    "total_entries": m.total_entries,
                    "avg_dwell_time": dwell_tracker.get_zone_average_dwell_time(name),
                }
                for name, m in zone_analyzer.get_all_metrics().items()
            }
            st.session_state.transition_matrix = transition_matrix.get_matrix().tolist()
            st.session_state.zone_names = zone_names

        results["hourly_distribution"] = traffic_counter.get_hourly_distribution()
        results["video_path"] = video_path
        results["width"] = metadata["width"]
        results["height"] = metadata["height"]
        st.session_state.processed = True
        st.session_state.results = results
        st.success(
            f"Processed {results['total_frames']} frames, "
            f"detected {results['unique_persons']} unique persons"
        )


def _traffic_overview_tab() -> None:
    """Render the traffic overview tab with key metrics and charts."""
    st.header("Traffic Overview")

    if not st.session_state.processed:
        st.info("Upload and process a video first")
        return

    results = st.session_state.results
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Visitors", results["unique_persons"])
    col2.metric("Total Detections", results["detection_count"])
    col3.metric("Total Frames", results["total_frames"])

    st.subheader("Hourly Distribution")
    hourly = results.get("hourly_distribution", {})
    if hourly:
        df = pd.DataFrame(list(hourly.items()), columns=["Hour", "Visitors"])
        fig = px.bar(df, x="Hour", y="Visitors", title="Visitors by Hour")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hourly data available")


def _zone_analysis_tab() -> None:
    """Render the zone analysis tab with per-zone metrics and transitions."""
    st.header("Zone Analysis")

    if not st.session_state.processed:
        st.info("Upload and process a video first")
        return

    zone_metrics = st.session_state.zone_metrics
    if zone_metrics:
        rows = [
            {
                "Zone": name,
                "Visitors": m["unique_visitors"],
                "Entries": m["total_entries"],
                "Avg Dwell Time (s)": round(m["avg_dwell_time"], 2),
            }
            for name, m in zone_metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Zone Transitions")
        matrix = st.session_state.transition_matrix
        zone_names = st.session_state.zone_names
        if matrix and zone_names:
            fig = px.imshow(
                matrix,
                labels={"x": "To Zone", "y": "From Zone", "color": "Transitions"},
                x=zone_names,
                y=zone_names,
                title="Zone Transition Matrix",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a zones configuration file to see zone analysis")


def _heatmap_view_tab() -> None:
    """Render the heatmap visualization tab."""
    st.header("Heatmap View")

    if not st.session_state.processed:
        st.info("Upload and process a video first")
        return

    results = st.session_state.results
    trajectories = results.get("trajectories", {})

    if not trajectories:
        st.warning("No trajectories available for heatmap generation")
        return

    col1, col2 = st.columns(2)
    with col1:
        sigma = st.slider("Smoothing (sigma)", 5.0, 50.0, 20.0, 1.0)
    with col2:
        colormap_name = st.selectbox(
            "Colormap",
            ["JET", "HOT", "INFERNO", "TURBO"],
        )

    colormap_map = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "TURBO": cv2.COLORMAP_TURBO,
    }
    colormap = colormap_map[colormap_name]

    width = results.get("width", 640)
    height = results.get("height", 480)

    from src.analytics.heatmap import HeatmapGenerator

    generator = HeatmapGenerator.from_trajectories(
        trajectories, width, height, sigma=sigma
    )
    colored = generator.generate_colored_heatmap(colormap=colormap)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Detection Heatmap", use_container_width=True)


def _video_playback_tab() -> None:
    """Render the annotated video playback tab."""
    st.header("Video Playback")

    if not st.session_state.processed:
        st.info("Upload and process a video first")
        return

    results = st.session_state.results
    video_path = results.get("video_path")

    if video_path:
        st.video(video_path)
        st.caption(
            f"Processed {results['total_frames']} frames | "
            f"{results['unique_persons']} unique persons detected"
        )
    else:
        st.info("No video path available for playback")

    trajectories = results.get("trajectories", {})
    if trajectories:
        st.subheader("Trajectory Summary")
        rows = [
            {
                "Person ID": tid,
                "Points Tracked": len(pts),
            }
            for tid, pts in trajectories.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
