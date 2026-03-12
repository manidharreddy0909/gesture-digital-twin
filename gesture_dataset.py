"""
Gesture Dataset Recording & Playback Module

Responsibilities:
- Record live hand landmark sequences with gesture labels
- Store datasets to JSON and CSV formats
- Replay recordings in GUI for training/verification
- Dataset statistics and management
- Easy export for ML training

Architecture:
- GestureRecord: Single gesture recording with metadata
- GestureDataset: Collection of gesture records
- DatasetRecorder: Records live gestures from tracking loop
- DatasetPlayer: Playback recorded sequences
- DatasetManager: I/O and dataset organization
"""

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from hand_tracker import HandLandmarks
from motion_analyzer import HandMotionInfo


@dataclass
class LandmarkRecord:
    """Single landmark point with metadata."""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class GestureFrame:
    """Single frame in a gesture recording."""
    timestamp: float
    frame_index: int
    left_hand: Optional[List[LandmarkRecord]] = None
    right_hand: Optional[List[LandmarkRecord]] = None
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0


@dataclass
class GestureRecord:
    """
    Complete recording of a single gesture instance.

    Attributes:
        gesture_name: Label of the gesture (e.g., "pinch", "swipe_left")
        frames: Sequence of hand landmarks over time
        duration_ms: Total duration in milliseconds
        hand_used: "left", "right", or "both"
        recording_date: ISO format datetime
        user_id: Optional user identifier
        camera_distance_cm: Estimated camera distance
        hand_size_normalized: Normalized hand bounding box size
        metadata: Additional custom metadata
    """
    gesture_name: str
    frames: List[GestureFrame] = field(default_factory=list)
    duration_ms: float = 0.0
    hand_used: str = "right"  # "left", "right", or "both"
    recording_date: str = field(default_factory=lambda: datetime.now().isoformat())
    user_id: str = "default"
    camera_distance_cm: float = 50.0
    hand_size_normalized: float = 0.0
    # Backward-compatible aliases used by older tests/examples.
    frame_count: int = 0
    duration_sec: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize legacy constructor args to canonical fields."""
        if self.duration_ms <= 0.0 and self.duration_sec > 0.0:
            self.duration_ms = self.duration_sec * 1000.0
        if not self.recording_date:
            self.recording_date = datetime.now().isoformat()

    def num_frames(self) -> int:
        """Get number of frames in recording."""
        if self.frames:
            return len(self.frames)
        return max(0, int(self.frame_count))

    def fps(self) -> float:
        """Calculate frames per second."""
        if self.duration_ms == 0:
            return 0.0
        return (self.num_frames() - 1) / (self.duration_ms / 1000.0) if self.duration_ms > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gesture_name": self.gesture_name,
            "duration_ms": self.duration_ms,
            "hand_used": self.hand_used,
            "recording_date": self.recording_date,
            "user_id": self.user_id,
            "camera_distance_cm": self.camera_distance_cm,
            "hand_size_normalized": self.hand_size_normalized,
            "num_frames": self.num_frames(),
            "fps": self.fps(),
            "metadata": self.metadata,
            "frame_count": self.num_frames(),
            "duration_sec": self.duration_ms / 1000.0 if self.duration_ms > 0 else 0.0,
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "frame_index": f.frame_index,
                    "left_hand": [asdict(lm) for lm in f.left_hand] if f.left_hand else None,
                    "right_hand": [asdict(lm) for lm in f.right_hand] if f.right_hand else None,
                    "left_hand_confidence": f.left_hand_confidence,
                    "right_hand_confidence": f.right_hand_confidence,
                }
                for f in self.frames
            ],
        }

    @staticmethod
    def from_dict(data: Dict) -> GestureRecord:
        """Create GestureRecord from dictionary."""
        frames = []
        for f_data in data.get("frames", []):
            left_hand = None
            if f_data.get("left_hand"):
                left_hand = [LandmarkRecord(**lm) for lm in f_data["left_hand"]]

            right_hand = None
            if f_data.get("right_hand"):
                right_hand = [LandmarkRecord(**lm) for lm in f_data["right_hand"]]

            frames.append(
                GestureFrame(
                    timestamp=f_data["timestamp"],
                    frame_index=f_data["frame_index"],
                    left_hand=left_hand,
                    right_hand=right_hand,
                    left_hand_confidence=f_data.get("left_hand_confidence", 0.0),
                    right_hand_confidence=f_data.get("right_hand_confidence", 0.0),
                )
            )

        return GestureRecord(
            gesture_name=data["gesture_name"],
            frames=frames,
            duration_ms=data["duration_ms"],
            hand_used=data["hand_used"],
            recording_date=data["recording_date"],
            user_id=data.get("user_id", "default"),
            camera_distance_cm=data.get("camera_distance_cm", 50.0),
            hand_size_normalized=data.get("hand_size_normalized", 0.0),
            frame_count=int(data.get("frame_count", 0)),
            duration_sec=float(data.get("duration_sec", 0.0)),
            metadata=data.get("metadata", {}),
        )


class GestureDataset:
    """Collection and management of gesture recordings."""

    def __init__(self):
        self.records: Dict[str, List[GestureRecord]] = {}  # gesture_name -> list of records
        self.total_duration_ms: float = 0.0
        self.created: str = datetime.now().isoformat()

    def add_record(self, record: GestureRecord) -> None:
        """Add a gesture record to dataset."""
        if record.gesture_name not in self.records:
            self.records[record.gesture_name] = []
        self.records[record.gesture_name].append(record)
        self.total_duration_ms += record.duration_ms

    def get_records(self, gesture_name: str) -> List[GestureRecord]:
        """Get all recordings of a specific gesture."""
        return self.records.get(gesture_name, [])

    def get_gesture_names(self) -> List[str]:
        """Get list of all gesture types in dataset."""
        return list(self.records.keys())

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_gestures": sum(len(recs) for recs in self.records.values()),
            "unique_gesture_types": len(self.records),
            "total_duration_ms": self.total_duration_ms,
            "created": self.created,
            "gestures": {},
        }

        for gesture_name, records in self.records.items():
            stats["gestures"][gesture_name] = {
                "count": len(records),
                "avg_duration_ms": sum(r.duration_ms for r in records) / len(records),
                "min_duration_ms": min(r.duration_ms for r in records),
                "max_duration_ms": max(r.duration_ms for r in records),
                "avg_fps": sum(r.fps() for r in records) / len(records),
            }

        return stats

    def to_dict(self) -> Dict:
        """Convert dataset to dictionary."""
        return {
            "created": self.created,
            "total_duration_ms": self.total_duration_ms,
            "records": {
                gesture_name: [record.to_dict() for record in records]
                for gesture_name, records in self.records.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict) -> GestureDataset:
        """Create dataset from dictionary."""
        dataset = GestureDataset()
        dataset.created = data.get("created", datetime.now().isoformat())

        for gesture_name, records_data in data.get("records", {}).items():
            for record_data in records_data:
                record = GestureRecord.from_dict(record_data)
                dataset.add_record(record)

        return dataset


class DatasetRecorder:
    """
    Record live hand tracking data as gesture datasets.

    Usage:
        recorder = DatasetRecorder()

        # In main loop:
        if user_starts_recording:
            recorder.start_recording("pinch", hand_used="left")

        # For each frame:
        recorder.record_frame(hands, motions)

        if user_stops_recording:
            record = recorder.stop_recording()
            dataset.add_record(record)
    """

    def __init__(self):
        self.recording: bool = False
        self.gesture_name: str = ""
        self.hand_used: str = ""
        self.frames: List[GestureFrame] = []
        self.start_time: float = 0.0
        self.camera_distance_cm: float = 50.0
        self.user_id: str = "default"
        self.frame_index: int = 0

    def start_recording(
        self,
        gesture_name: str,
        hand_used: str = "left",
        user_id: str = "default",
        camera_distance_cm: float = 50.0,
    ) -> None:
        """Start recording a new gesture."""
        self.recording = True
        self.gesture_name = gesture_name
        self.hand_used = hand_used
        self.user_id = user_id
        self.camera_distance_cm = camera_distance_cm
        self.frames = []
        self.start_time = time.time()
        self.frame_index = 0

    def record_frame(
        self,
        hands: List[HandLandmarks],
        timestamp: float,
    ) -> None:
        """Record a frame of hand data."""
        if not self.recording:
            return

        left_hand = None
        right_hand = None
        left_confidence = 0.0
        right_confidence = 0.0

        # Extract hand data based on hand_used setting
        for hand in hands:
            if hand.handedness.lower() == "left":
                left_hand = [
                    LandmarkRecord(
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        confidence=float(lm.z),  # z contains confidence
                    )
                    for lm in hand.landmarks
                ]
                left_confidence = 1.0  # Detected
            elif hand.handedness.lower() == "right":
                right_hand = [
                    LandmarkRecord(
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        confidence=float(lm.z),
                    )
                    for lm in hand.landmarks
                ]
                right_confidence = 1.0

        frame = GestureFrame(
            timestamp=timestamp,
            frame_index=self.frame_index,
            left_hand=left_hand if "left" in self.hand_used.lower() or self.hand_used == "both" else None,
            right_hand=right_hand if "right" in self.hand_used.lower() or self.hand_used == "both" else None,
            left_hand_confidence=left_confidence,
            right_hand_confidence=right_confidence,
        )
        self.frames.append(frame)
        self.frame_index += 1

    def stop_recording(self) -> Optional[GestureRecord]:
        """Stop recording and return the gesture record."""
        if not self.recording:
            return None

        self.recording = False
        duration_ms = (time.time() - self.start_time) * 1000.0

        # Calculate hand size (bounding box of first valid frame)
        hand_size = 0.0
        for frame in self.frames:
            hand = frame.left_hand or frame.right_hand
            if hand:
                xs = [lm.x for lm in hand]
                ys = [lm.y for lm in hand]
                hand_size = (max(xs) - min(xs)) + (max(ys) - min(ys))
                break

        record = GestureRecord(
            gesture_name=self.gesture_name,
            frames=self.frames,
            duration_ms=duration_ms,
            hand_used=self.hand_used,
            recording_date=datetime.now().isoformat(),
            user_id=self.user_id,
            camera_distance_cm=self.camera_distance_cm,
            hand_size_normalized=hand_size,
        )

        return record

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording


class DatasetManager:
    """
    I/O and persistence for gesture datasets.

    Supports:
    - JSON format (full fidelity)
    - CSV format (landmark-only, simplified)
    - Dataset metadata and versioning
    """

    def __init__(self, storage_dir: str = "datasets"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_json(self, dataset: GestureDataset, filename: str) -> Path:
        """Save dataset to JSON file."""
        file_path = self.storage_dir / f"{filename}.json"

        with open(file_path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2)

        print(f"[DatasetManager] Saved dataset to {file_path}")
        return file_path

    def load_json(self, filename: str) -> GestureDataset:
        """Load dataset from JSON file."""
        file_path = self.storage_dir / f"{filename}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        dataset = GestureDataset.from_dict(data)
        print(f"[DatasetManager] Loaded dataset from {file_path}")
        return dataset

    def export_csv(self, dataset: GestureDataset, filename: str) -> Path:
        """Export dataset to CSV file (simplified format)."""
        file_path = self.storage_dir / f"{filename}.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "gesture_name", "frame_index", "timestamp", "hand_used",
                "lm0_x", "lm0_y", "lm1_x", "lm1_y", "lm2_x", "lm2_y",  # Abbreviated for brevity
                "user_id", "date"
            ])

            # Data rows
            for gesture_name, records in dataset.records.items():
                for record in records:
                    for frame in record.frames[:10]:  # Limit exported frames
                        hand = frame.left_hand or frame.right_hand
                        if hand:
                            # Extract first 3 landmarks for CSV
                            lm_data = []
                            for i in range(3):
                                if i < len(hand):
                                    lm_data.extend([hand[i].x, hand[i].y])
                                else:
                                    lm_data.extend([0.0, 0.0])

                            writer.writerow([
                                gesture_name,
                                frame.frame_index,
                                frame.timestamp,
                                record.hand_used,
                                *lm_data,
                                record.user_id,
                                record.recording_date
                            ])

        print(f"[DatasetManager] Exported dataset to CSV: {file_path}")
        return file_path

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def get_dataset_info(self, filename: str) -> Dict:
        """Get metadata about a dataset without full load."""
        dataset = self.load_json(filename)
        return dataset.get_statistics()


class DatasetPlayer:
    """
    Playback recorded gesture sequences.

    Useful for:
    - Visualization in GUI
    - Debugging gesture detection
    - Training verification
    - Benchmarking
    """

    def __init__(self):
        self.current_dataset: Optional[GestureDataset] = None
        self.current_gesture_idx: int = 0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.playback_speed: float = 1.0  # 1.0 = normal, 2.0 = 2x

    def load_dataset(self, dataset: GestureDataset) -> None:
        """Load a dataset for playback."""
        self.current_dataset = dataset
        self.current_gesture_idx = 0
        self.current_frame_idx = 0

    def play(self) -> None:
        """Start playback."""
        if self.current_dataset is None:
            return
        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def next_frame(self) -> Optional[GestureFrame]:
        """Get next frame in playback."""
        if not self.is_playing or self.current_dataset is None:
            return None

        gestures = list(self.current_dataset.records.keys())
        if self.current_gesture_idx >= len(gestures):
            self.is_playing = False
            return None

        gesture_name = gestures[self.current_gesture_idx]
        records = self.current_dataset.get_records(gesture_name)

        # Iterate through all records and frames
        frame_count = 0
        for record in records:
            record_frame_count = len(record.frames)
            if self.current_frame_idx < frame_count + record_frame_count:
                return record.frames[self.current_frame_idx - frame_count]
            frame_count += record_frame_count

        # Move to next gesture
        self.current_gesture_idx += 1
        self.current_frame_idx = 0
        return self.next_frame()

    def reset(self) -> None:
        """Reset playback to beginning."""
        self.current_gesture_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
