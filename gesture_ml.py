"""
Machine Learning gesture recognition framework for advanced gesture detection.

This module provides:
- CNN-based static gesture recognition from landmark positions or hand images
- RNN/LSTM-based dynamic gesture recognition from trajectory histories
- Training hooks for easy custom gesture model integration
- Model management and caching
- Fallback to heuristic gestures when ML models aren't available

Architecture:
- StaticGestureModel: Base class for static gesture recognition
- DynamicGestureModel: Base class for dynamic gesture recognition
- CNNStaticGestureModel: CNN-based implementation
- LSTMDynamicGestureModel: LSTM-based implementation
- GestureModelManager: Manages model lifecycle and training
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
from pathlib import Path

import numpy as np

from motion_analyzer import HandMotionInfo, FingerTrajectory


@dataclass
class MLGestureResult:
    """Result from ML-based gesture recognition."""

    label: str
    confidence: float
    model_name: str
    inference_time_ms: float = 0.0


class StaticGestureModel(ABC):
    """
    Base class for static gesture recognition models.

    Static gestures are recognition tasks based on hand pose/shape at a single moment.
    """

    def __init__(self, model_name: str = "static_gesture_model"):
        self.model_name = model_name
        self.trained = False

    @abstractmethod
    def predict(self, landmarks: List) -> Tuple[str, float]:
        """
        Predict static gesture from hand landmarks.

        Args:
            landmarks: List of mediapipe landmarks (21 points, x/y/z normalized)

        Returns:
            (label, confidence) tuple
        """
        pass

    @abstractmethod
    def train(self, training_data: List[Tuple[List, str]]) -> None:
        """
        Train the model on labeled landmark data.

        Args:
            training_data: List of (landmarks, gesture_label) tuples
        """
        pass

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> StaticGestureModel:
        """Deserialize model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class DynamicGestureModel(ABC):
    """
    Base class for dynamic gesture recognition models.

    Dynamic gestures are temporal patterns detected from hand motion trajectories.
    """

    def __init__(self, model_name: str = "dynamic_gesture_model"):
        self.model_name = model_name
        self.trained = False

    @abstractmethod
    def predict(self, motion: HandMotionInfo) -> Tuple[str, float]:
        """
        Predict dynamic gesture from hand motion.

        Args:
            motion: HandMotionInfo containing fingertip trajectories

        Returns:
            (label, confidence) tuple
        """
        pass

    @abstractmethod
    def train(self, training_data: List[Tuple[HandMotionInfo, str]]) -> None:
        """
        Train the model on labeled motion data.

        Args:
            training_data: List of (motion_info, gesture_label) tuples
        """
        pass

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> DynamicGestureModel:
        """Deserialize model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class SimpleStaticGestureModel(StaticGestureModel):
    """
    Placeholder CNN-based static gesture model (easily replaceable with real CNN).

    Features extracted: landmark distances, angles, normalized positions.
    This is a simple nearest-neighbor classifier for demonstration.
    """

    def __init__(self, model_name: str = "cnn_static_gesture"):
        super().__init__(model_name)
        self.training_samples: List[Tuple[np.ndarray, str]] = []
        self.gesture_labels: List[str] = []

    def _extract_features(self, landmarks: List) -> np.ndarray:
        """Extract hand features from landmarks for ML model."""
        features = []

        # Normalize all landmarks relative to wrist (landmark 0)
        wrist = (landmarks[0].x, landmarks[0].y)
        for lm in landmarks:
            x = lm.x - wrist[0]
            y = lm.y - wrist[1]
            features.extend([x, y])

        # Add simple pairwise distances (thumb-index, index-middle, etc.)
        for i in range(0, 20):
            for j in range(i + 1, min(i + 5, 21)):
                dx = landmarks[i].x - landmarks[j].x
                dy = landmarks[i].y - landmarks[j].y
                dist = (dx * dx + dy * dy) ** 0.5
                features.append(dist)

        return np.array(features, dtype=np.float32)

    def predict(self, landmarks: List) -> Tuple[str, float]:
        """Predict using nearest neighbor in feature space."""
        if not self.trained or len(self.training_samples) == 0:
            return "unknown", 0.0

        features = self._extract_features(landmarks)
        min_dist = float('inf')
        best_label = "unknown"
        best_conf = 0.0

        for train_features, label in self.training_samples:
            dist = np.linalg.norm(features - train_features)
            if dist < min_dist:
                min_dist = dist
                best_label = label
                # Confidence inversely proportional to distance
                best_conf = max(0.0, 1.0 - min(min_dist / 100.0, 1.0))

        return best_label, best_conf

    def train(self, training_data: List[Tuple[List, str]]) -> None:
        """Train on labeled landmark data."""
        self.training_samples = []
        self.gesture_labels = []

        for landmarks, label in training_data:
            features = self._extract_features(landmarks)
            self.training_samples.append((features, label))
            if label not in self.gesture_labels:
                self.gesture_labels.append(label)

        self.trained = True


class SimpleDynamicGestureModel(DynamicGestureModel):
    """
    Placeholder LSTM-based dynamic gesture model (easily replaceable with real LSTM).

    Features from trajectories: velocity profiles, path curvature, motion direction.
    """

    def __init__(self, model_name: str = "lstm_dynamic_gesture"):
        super().__init__(model_name)
        self.training_samples: List[Tuple[np.ndarray, str]] = []
        self.gesture_labels: List[str] = []

    def _extract_trajectory_features(self, motion: HandMotionInfo, max_len: int = 20) -> np.ndarray:
        """Extract features from fingertip trajectories."""
        features = []

        # Get index fingertip trajectory (primary for gesture)
        traj = motion.fingertip_trajectories.get(8)
        if traj is None:
            return np.zeros(max_len * 4, dtype=np.float32)

        # Pad/truncate trajectory to fixed length
        speeds = list(traj.speeds)[:max_len]
        speeds.extend([0.0] * (max_len - len(speeds)))

        # Direction angles (if available)
        angles = []
        for i in range(len(traj.points) - 1):
            x0, y0, _ = traj.points[i]
            x1, y1, _ = traj.points[i + 1]
            angle = np.arctan2(y1 - y0, x1 - x0)
            angles.append(angle)

        angles.extend([0.0] * (max_len - len(angles)))
        angles = angles[:max_len]

        # Normalize and flatten
        features.extend(speeds)
        features.extend(angles)
        features.append(traj.path_length())
        features.append(traj.net_displacement())

        return np.array(features, dtype=np.float32)

    def predict(self, motion: HandMotionInfo) -> Tuple[str, float]:
        """Predict using nearest neighbor in trajectory feature space."""
        if not self.trained or len(self.training_samples) == 0:
            return "unknown", 0.0

        features = self._extract_trajectory_features(motion)
        min_dist = float('inf')
        best_label = "unknown"
        best_conf = 0.0

        for train_features, label in self.training_samples:
            dist = np.linalg.norm(features - train_features)
            if dist < min_dist:
                min_dist = dist
                best_label = label
                best_conf = max(0.0, 1.0 - min(min_dist / 100.0, 1.0))

        return best_label, best_conf

    def train(self, training_data: List[Tuple[HandMotionInfo, str]]) -> None:
        """Train on labeled motion data."""
        self.training_samples = []
        self.gesture_labels = []

        for motion, label in training_data:
            features = self._extract_trajectory_features(motion)
            self.training_samples.append((features, label))
            if label not in self.gesture_labels:
                self.gesture_labels.append(label)

        self.trained = True


class GestureModelManager:
    """
    Manages lifecycle of static and dynamic gesture models.

    Responsibilities:
    - Load/save models from disk
    - Manage training data collection
    - Integrate predictions with heuristic fallback
    - Support multiple gesture types
    """

    def __init__(self, model_dir: str = "models/gestures"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.static_model: Optional[StaticGestureModel] = None
        self.dynamic_model: Optional[DynamicGestureModel] = None

        self.training_data_static: List[Tuple[List, str]] = []
        self.training_data_dynamic: List[Tuple[HandMotionInfo, str]] = []

    def initialize_default_models(self) -> None:
        """Create and initialize default ML models."""
        self.static_model = SimpleStaticGestureModel()
        self.dynamic_model = SimpleDynamicGestureModel()

    def load_static_model(self, model_name: str) -> bool:
        """Load a static gesture model from disk."""
        path = self.model_dir / f"{model_name}.pkl"
        try:
            self.static_model = StaticGestureModel.load(str(path))
            return True
        except Exception as e:
            print(f"Failed to load static model: {e}")
            return False

    def save_static_model(self, model_name: str) -> bool:
        """Save the current static gesture model."""
        if self.static_model is None:
            return False
        path = self.model_dir / f"{model_name}.pkl"
        try:
            self.static_model.save(str(path))
            return True
        except Exception as e:
            print(f"Failed to save static model: {e}")
            return False

    def load_dynamic_model(self, model_name: str) -> bool:
        """Load a dynamic gesture model from disk."""
        path = self.model_dir / f"{model_name}.pkl"
        try:
            self.dynamic_model = DynamicGestureModel.load(str(path))
            return True
        except Exception as e:
            print(f"Failed to load dynamic model: {e}")
            return False

    def save_dynamic_model(self, model_name: str) -> bool:
        """Save the current dynamic gesture model."""
        if self.dynamic_model is None:
            return False
        path = self.model_dir / f"{model_name}.pkl"
        try:
            self.dynamic_model.save(str(path))
            return True
        except Exception as e:
            print(f"Failed to save dynamic model: {e}")
            return False

    def collect_static_training_sample(self, landmarks: List, label: str) -> None:
        """Collect a labeled static gesture sample for training."""
        self.training_data_static.append((landmarks, label))

    def collect_dynamic_training_sample(self, motion: HandMotionInfo, label: str) -> None:
        """Collect a labeled dynamic gesture sample for training."""
        self.training_data_dynamic.append((motion, label))

    def train_static_model(self) -> bool:
        """Train the static gesture model on collected data."""
        if self.static_model is None or len(self.training_data_static) == 0:
            return False

        try:
            self.static_model.train(self.training_data_static)
            return True
        except Exception as e:
            print(f"Failed to train static model: {e}")
            return False

    def train_dynamic_model(self) -> bool:
        """Train the dynamic gesture model on collected data."""
        if self.dynamic_model is None or len(self.training_data_dynamic) == 0:
            return False

        try:
            self.dynamic_model.train(self.training_data_dynamic)
            return True
        except Exception as e:
            print(f"Failed to train dynamic model: {e}")
            return False

    def predict_static(self, landmarks: List) -> Optional[MLGestureResult]:
        """Get ML prediction for static gesture."""
        if self.static_model is None or not self.static_model.trained:
            return None

        try:
            label, conf = self.static_model.predict(landmarks)
            return MLGestureResult(
                label=label,
                confidence=conf,
                model_name=self.static_model.model_name,
            )
        except Exception as e:
            print(f"Error in static gesture prediction: {e}")
            return None

    def predict_dynamic(self, motion: HandMotionInfo) -> Optional[MLGestureResult]:
        """Get ML prediction for dynamic gesture."""
        if self.dynamic_model is None or not self.dynamic_model.trained:
            return None

        try:
            label, conf = self.dynamic_model.predict(motion)
            return MLGestureResult(
                label=label,
                confidence=conf,
                model_name=self.dynamic_model.model_name,
            )
        except Exception as e:
            print(f"Error in dynamic gesture prediction: {e}")
            return None

    def get_training_data_counts(self) -> Dict[str, int]:
        """Get counts of collected training samples by label."""
        static_counts = {}
        for _, label in self.training_data_static:
            static_counts[label] = static_counts.get(label, 0) + 1

        dynamic_counts = {}
        for _, label in self.training_data_dynamic:
            dynamic_counts[label] = dynamic_counts.get(label, 0) + 1

        return {"static": static_counts, "dynamic": dynamic_counts}
