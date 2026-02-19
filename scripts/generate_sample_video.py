"""Generate a synthetic sample video for testing and demonstration.

Creates a short video with moving rectangles simulating person
movement patterns in a retail environment.
"""

from pathlib import Path

import cv2
import numpy as np


def generate_sample_video(
    output_path: str = "data/sample/demo.mp4",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    duration_seconds: float = 5.0,
) -> str:
    """Generate a synthetic sample video with moving objects.

    Args:
        output_path: Path for the output video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        duration_seconds: Video duration in seconds.

    Returns:
        Path to the generated video file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(fps * duration_seconds)
    rng = np.random.RandomState(42)

    persons = [
        {"x": 50, "y": 200, "dx": 3, "dy": 1},
        {"x": 400, "y": 100, "dx": -2, "dy": 2},
        {"x": 300, "y": 350, "dx": 1, "dy": -1},
    ]

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)

        cv2.rectangle(frame, (80, 80), (280, 380), (0, 200, 0), 2)
        cv2.putText(
            frame, "Entrance", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
        )

        cv2.rectangle(frame, (400, 150), (600, 400), (0, 0, 200), 2)
        cv2.putText(
            frame, "Checkout", (420, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1
        )

        for i, p in enumerate(persons):
            p["x"] += p["dx"] + rng.randint(-1, 2)
            p["y"] += p["dy"] + rng.randint(-1, 2)
            p["x"] = max(20, min(width - 40, p["x"]))
            p["y"] = max(20, min(height - 60, p["y"]))

            cv2.rectangle(
                frame,
                (p["x"], p["y"]),
                (p["x"] + 30, p["y"] + 50),
                (255, 100, 50),
                -1,
            )
            cv2.putText(
                frame,
                f"P{i + 1}",
                (p["x"] + 5, p["y"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            frame,
            f"Frame {frame_idx + 1}/{total_frames}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1,
        )

        writer.write(frame)

    writer.release()
    return output_path


if __name__ == "__main__":
    path = generate_sample_video()
    print(f"Sample video generated: {path}")
