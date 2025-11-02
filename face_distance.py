"""Webcam face tracker with distance, gender, and age overlays.

This script uses MediaPipe's Face Mesh solution to detect facial landmarks,
draws a bounding box around each detected face, overlays a magenta crosshair on
the nose tip, and estimates the distance from the camera in centimeters using a
default camera field-of-view assumption. For each detected face it also
leverages DeepFace's high-accuracy models to infer gender and age, displaying
the predictions beneath the bounding box.

Press "q" to quit the application.
"""

from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


# MediaPipe Face Mesh landmark index for the nose tip.
NOSE_TIP_ID = 1


@dataclass
class DetectionResult:
    """Holds the information required for drawing overlays on a video frame."""

    bbox: Tuple[int, int, int, int]
    distance_cm: Optional[float]
    nose: Tuple[int, int]
    gender: Optional[str] = None
    age: Optional[int] = None


class AgeGenderEstimator:
    """Estimates gender and age for detected faces using DeepFace."""

    def __init__(self, sample_every_n_frames: int = 10, margin_ratio: float = 0.25) -> None:
        self.sample_every_n_frames = sample_every_n_frames
        self.margin_ratio = margin_ratio
        self._frame_counter = 0
        self._cached_predictions: List[Tuple[Optional[str], Optional[int]]] = []
        self._deepface = None
        self._warning_emitted = False
        self._deepface_available = True

    def _ensure_model(self) -> None:
        if self._deepface is not None or not self._deepface_available:
            return

        try:
            deepface_module = importlib.import_module("deepface")
        except ModuleNotFoundError:
            self._log_warning(
                "DeepFace paketi bulunamadı. Lütfen `python run_face_distance.py`"
                " komutunu çalıştırarak bağımlılıkları otomatik kurun veya"
                " manuel olarak `pip install deepface` komutunu yürütün."
            )
            self._deepface_available = False
            return
        except ImportError as exc:
            self._log_warning(
                "DeepFace kütüphanesi yüklenirken hata oluştu. Lütfen kurulumun"
                " tamamlandığından emin olun. Ayrıntı:" f" {exc}"
            )
            self._deepface_available = False
            return

        deepface_class = getattr(deepface_module, "DeepFace", None)
        if deepface_class is None:
            self._log_warning(
                "DeepFace kütüphanesi yüklü fakat `DeepFace` sınıfı bulunamadı."
                " Kurulumu yeniden yapmayı deneyin."
            )
            self._deepface_available = False
            return

        self._deepface = deepface_class

    def enrich(self, frame: np.ndarray, detections: Sequence[DetectionResult]) -> None:
        self._frame_counter += 1

        if not detections:
            self._cached_predictions = []
            return

        should_refresh = (
            self._frame_counter % self.sample_every_n_frames == 0
            or len(self._cached_predictions) != len(detections)
        )

        if should_refresh:
            self._ensure_model()
            predictions: List[Tuple[Optional[str], Optional[int]]] = []

            if not self._deepface_available or self._deepface is None:
                predictions = [(None, None)] * len(detections)
            else:
                for detection in detections:
                    face_image = self._extract_face(frame, detection.bbox)
                    if face_image.size == 0:
                        predictions.append((None, None))
                        continue

                    try:
                        analysis = self._deepface.analyze(
                            face_image,
                            actions=("gender", "age"),
                            enforce_detection=False,
                            detector_backend="skip",
                            prog_bar=False,
                        )
                    except Exception as exc:
                        self._log_warning(
                            "DeepFace analizinde hata oluştu. Lütfen TensorFlow ve DeepFace"
                            " bağımlılıklarının tam kurulduğunu kontrol edin. Ayrıntı:"
                            f" {exc}"
                        )
                        predictions.append((None, None))
                        continue

                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    gender_label = self._parse_gender(analysis)
                    age_estimate = self._parse_age(analysis)
                    predictions.append((gender_label, age_estimate))

            self._cached_predictions = predictions

        for detection, (gender, age) in zip(detections, self._cached_predictions):
            detection.gender = gender
            detection.age = age

    def _log_warning(self, message: str) -> None:
        if self._warning_emitted:
            return
        print(f"[AgeGenderEstimator] {message}")
        self._warning_emitted = True

    def _extract_face(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        min_x, min_y, max_x, max_y = bbox

        face_width = max_x - min_x
        face_height = max_y - min_y

        margin_x = int(face_width * self.margin_ratio)
        margin_y = int(face_height * self.margin_ratio)

        x1 = max(min_x - margin_x, 0)
        y1 = max(min_y - margin_y, 0)
        x2 = min(max_x + margin_x, width)
        y2 = min(max_y + margin_y, height)

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        cropped = frame[y1:y2, x1:x2]
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    def _parse_gender(self, analysis: dict) -> Optional[str]:
        label = analysis.get("dominant_gender")
        if not label and isinstance(analysis.get("gender"), dict):
            gender_scores = analysis["gender"]
            label = max(gender_scores, key=gender_scores.get)

        if not label:
            return None

        normalized = label.lower()
        if normalized == "woman":
            return "Kadın"
        if normalized == "man":
            return "Erkek"

        return label

    def _parse_age(self, analysis: dict) -> Optional[int]:
        age_value = analysis.get("age")
        if age_value is None and "age" in analysis and isinstance(analysis["age"], dict):
            age_dict = analysis["age"]
            age_value = max(age_dict, key=age_dict.get)

        if age_value is None:
            return None

        try:
            return int(round(float(age_value)))
        except (TypeError, ValueError):
            return None


class FaceDistanceEstimator:
    """Tracks a face and estimates the distance from the camera."""

    def __init__(self, known_face_width_cm: float, horizontal_fov_degrees: float) -> None:
        self.known_face_width_cm = known_face_width_cm
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self._frame_width: Optional[int] = None

        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
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

    def process_frame(self, frame) -> List[DetectionResult]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return []

        height, width, _ = frame.shape
        self._frame_width = width
        detections = []

        for face_landmarks in result.multi_face_landmarks:
            pixel_landmarks = self._landmarks_to_pixels(face_landmarks.landmark, width, height)

            bbox = self._compute_bbox(pixel_landmarks)
            bbox_width = bbox[2] - bbox[0]
            distance_cm = self.estimate_distance(bbox_width)

            nose_point = tuple(pixel_landmarks[NOSE_TIP_ID])

            detections.append(
                DetectionResult(bbox=bbox, distance_cm=distance_cm, nose=nose_point)
            )

        return detections


def _draw_crosshair(
    frame,
    center: Tuple[int, int],
    *,
    color: Tuple[int, int, int] = (255, 0, 255),
    arm_length: int = 18,
    gap: int = 6,
    thickness: int = 4,
) -> None:
    """Draw a four-armed crosshair with a center gap."""

    x, y = center

    # Horizontal arms
    cv2.line(frame, (x - arm_length, y), (x - gap, y), color, thickness)
    cv2.line(frame, (x + gap, y), (x + arm_length, y), color, thickness)

    # Vertical arms
    cv2.line(frame, (x, y - arm_length), (x, y - gap), color, thickness)
    cv2.line(frame, (x, y + gap), (x, y + arm_length), color, thickness)


def draw_overlays(frame, detections: Iterable[DetectionResult]) -> None:
    frame_height, frame_width, _ = frame.shape

    for detection in detections:
        min_x, min_y, max_x, max_y = detection.bbox
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        label_position = (min_x, max(min_y - 10, 20))
        cv2.putText(frame, "Hedef", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _draw_crosshair(frame, detection.nose)

        if detection.distance_cm is not None:
            distance_text = f"{detection.distance_cm:.1f} cm"
        else:
            distance_text = "N/A"

        distance_size, distance_baseline = cv2.getTextSize(
            distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        distance_width, distance_height = distance_size
        distance_x = min(max_x - distance_width, frame_width - distance_width - 5)
        distance_x = max(distance_x, 5)
        desired_y = max_y + distance_height + 6
        distance_y = min(desired_y, frame_height - distance_baseline - 5)

        cv2.putText(
            frame,
            distance_text,
            (distance_x, distance_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        info_parts = []
        if detection.gender:
            info_parts.append(detection.gender)
        if detection.age is not None:
            info_parts.append(f"{detection.age} yaş")

        if info_parts:
            info_text = " · ".join(info_parts)
            info_size, info_baseline = cv2.getTextSize(
                info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            info_width, info_height = info_size
            info_x = min(max_x - info_width, frame_width - info_width - 5)
            info_x = max(info_x, 5)
            info_y = distance_y + info_height + info_baseline + 6
            info_y = min(info_y, frame_height - info_baseline - 5)
            cv2.putText(
                frame,
                info_text,
                (info_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    instructions = "Press 'q' to quit"
    cv2.putText(frame, instructions, (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


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
    age_gender_estimator = AgeGenderEstimator()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera with index {args.camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting.")
                break

            detections = estimator.process_frame(frame)
            age_gender_estimator.enrich(frame, detections)
            draw_overlays(frame, detections)

            cv2.imshow("Face Distance Estimator", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
