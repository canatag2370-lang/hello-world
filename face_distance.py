"""Webcam face tracker with facial landmarks and distance estimation.

This script uses MediaPipe's Face Mesh solution to detect facial landmarks,
draws a bounding box around the detected face, highlights key facial points,
and estimates the distance from the camera in centimeters using a default
camera field-of-view assumption.

Press "q" to quit the application.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


# MediaPipe Face Mesh landmark indices for key facial features.
LANDMARK_IDS: Dict[str, int] = {
    "nose_tip": 1,
    "left_cheek": 234,
    "right_cheek": 454,
    "chin": 152,
    "forehead": 10,
}


@dataclass
class DetectionResult:
    """Holds the information required for drawing overlays on a video frame."""

    bbox: Tuple[int, int, int, int]
    distance_cm: Optional[float]
    landmarks: Dict[str, Tuple[int, int]]


class FaceDistanceEstimator:
    """Tracks a face and estimates the distance from the camera."""

    def __init__(self, known_face_width_cm: float, horizontal_fov_degrees: float) -> None:
        self.known_face_width_cm = known_face_width_cm
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self._frame_width: Optional[int] = None

        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _landmarks_to_pixels(
        self, face_landmarks: Iterable[landmark_pb2.NormalizedLandmark], width: int, height: int
    ) -> np.ndarray:
        coords = np.array([(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks])
        return coords

    def _compute_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        min_x = int(np.min(landmarks[:, 0]))
        max_x = int(np.max(landmarks[:, 0]))
        min_y = int(np.min(landmarks[:, 1]))
        max_y = int(np.max(landmarks[:, 1]))
        return min_x, min_y, max_x, max_y

    def estimate_distance(self, bbox_width_pixels: float) -> Optional[float]:
        if bbox_width_pixels <= 0:
            return None

        focal_length = self._compute_focal_length_px()
        distance = (self.known_face_width_cm * focal_length) / bbox_width_pixels
        return distance

    def _compute_focal_length_px(self) -> float:
        """Compute focal length in pixels based on frame width and FOV."""

        if self._frame_width is None:
            raise RuntimeError("Frame width unavailable for focal length computation.")

        return (self._frame_width / 2) / math.tan(math.radians(self.horizontal_fov_degrees / 2))

    def process_frame(self, frame) -> Optional[DetectionResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return None

        height, width, _ = frame.shape
        self._frame_width = width
        face_landmarks = result.multi_face_landmarks[0].landmark
        pixel_landmarks = self._landmarks_to_pixels(face_landmarks, width, height)

        bbox = self._compute_bbox(pixel_landmarks)
        bbox_width = bbox[2] - bbox[0]
        distance_cm = self.estimate_distance(bbox_width)

        key_landmarks = {
            name: tuple(pixel_landmarks[idx]) for name, idx in LANDMARK_IDS.items()
        }

        return DetectionResult(bbox=bbox, distance_cm=distance_cm, landmarks=key_landmarks)


def draw_overlays(frame, detection: DetectionResult) -> None:
    min_x, min_y, max_x, max_y = detection.bbox
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    label_position = (min_x, max(min_y - 10, 20))
    cv2.putText(frame, "Hedef", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x, y) in detection.landmarks.values():
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    if detection.distance_cm:
        text = f"Distance: {detection.distance_cm:.1f} cm"
    else:
        text = "Distance unavailable"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    instructions = "Press 'q' to quit"
    cv2.putText(frame, instructions, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the webcam to use (default: 0)",
    )
    parser.add_argument(
        "--known-face-width",
        type=float,
        default=16.0,
        help="Approximate width of your face in centimeters (default: 16.0 cm)",
    )
    parser.add_argument(
        "--horizontal-fov",
        type=float,
        default=60.0,
        help="Approximate horizontal field of view of your camera in degrees (default: 60°)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator = FaceDistanceEstimator(args.known_face_width, args.horizontal_fov)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera with index {args.camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting.")
                break

            detection = estimator.process_frame(frame)
            if detection:
                draw_overlays(frame, detection)

            cv2.imshow("Face Distance Estimator", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
