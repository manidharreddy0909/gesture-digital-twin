import os
import time
import ctypes
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------- Hand model ----------------- #
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = "hand_landmarker.task"

def download_hand_landmarker_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading model to '{model_path}'...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Download complete.")
    return model_path

# ----------------- Hand connections ----------------- #
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ----------------- Cursor helpers ----------------- #
def _get_screen_size() -> Tuple[int, int]:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def _set_cursor_pos(x: int, y: int) -> None:
    ctypes.windll.user32.SetCursorPos(int(x), int(y))

# ----------------- Hand state ----------------- #
@dataclass
class HandState:
    smoothed_x: Optional[float] = None
    smoothed_y: Optional[float] = None
    motion_history: List[Tuple[float,float]] = field(default_factory=list)
    velocity: Tuple[float,float] = (0.0,0.0)
    gesture: Optional[str] = None

# ----------------- Tracker config ----------------- #
@dataclass
class HandTrackerConfig:
    model_path: str = DEFAULT_MODEL_PATH
    max_num_hands: int = 2
    smoothing_factor: float = 0.2
    mirror_image: bool = True
    enable_cursor_control: bool = True

# ----------------- HandTracker ----------------- #
class HandTracker:
    def __init__(self, config: HandTrackerConfig):
        self.config = config
        model_path = download_hand_landmarker_model(config.model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.screen_width, self.screen_height = _get_screen_size()
        self.hands = [HandState() for _ in range(config.max_num_hands)]
        self._start_time = time.time()

    def close(self):
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    def _timestamp_ms(self) -> int:
        return int((time.time()-self._start_time)*1000)

    def _update_cursor(self, hand_id:int, x_norm:float, y_norm:float) -> Tuple[int,int]:
        state = self.hands[hand_id]
        target_x = x_norm * (self.screen_width-1)
        target_y = y_norm * (self.screen_height-1)
        # Exponential smoothing
        if state.smoothed_x is None:
            state.smoothed_x, state.smoothed_y = target_x, target_y
        else:
            alpha = self.config.smoothing_factor
            state.smoothed_x = (1-alpha)*state.smoothed_x + alpha*target_x
            state.smoothed_y = (1-alpha)*state.smoothed_y + alpha*target_y
        # Velocity
        if state.motion_history:
            last_x,last_y = state.motion_history[-1]
            state.velocity = (state.smoothed_x-last_x, state.smoothed_y-last_y)
        state.motion_history.append((state.smoothed_x,state.smoothed_y))
        if len(state.motion_history)>5:
            state.motion_history.pop(0)
        # Cursor control (right hand = first hand)
        if self.config.enable_cursor_control and hand_id==0:
            _set_cursor_pos(int(state.smoothed_x),int(state.smoothed_y))
        return int(state.smoothed_x), int(state.smoothed_y)

    def detect_gesture(self, hand_id:int) -> str:
        state = self.hands[hand_id]
        if len(state.motion_history)<2:
            return "none"
        dx = state.motion_history[-1][0] - state.motion_history[0][0]
        dy = state.motion_history[-1][1] - state.motion_history[0][1]
        if abs(dx)>100 and abs(dx)>abs(dy): return "swipe_right" if dx>0 else "swipe_left"
        if abs(dy)>100: return "swipe_down" if dy>0 else "swipe_up"
        return "none"

    @staticmethod
    def _draw(frame:np.ndarray, landmarks_norm:List):
        h,w,_ = frame.shape
        points=[(int(lm.x*w),int(lm.y*h)) for lm in landmarks_norm]
        # Draw connections
        for start,end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0,255,0), 2)
        # Draw landmarks
        for i,(x,y) in enumerate(points):
            color=(0,0,255) if i==8 else (255,0,0)
            cv2.circle(frame,(x,y),4,color,-1)

    def process(self, frame_bgr:np.ndarray) -> np.ndarray:
        frame = frame_bgr.copy()
        if self.config.mirror_image: frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image,self._timestamp_ms())
        if result.hand_landmarks:
            for hand_id,hand_landmarks in enumerate(result.hand_landmarks):
                self._draw(frame,hand_landmarks)
                index_tip = hand_landmarks[8]
                cursor_pos = self._update_cursor(hand_id,index_tip.x,index_tip.y)
                gesture = self.detect_gesture(hand_id)
                self.hands[hand_id].gesture = gesture
                cv2.putText(frame,f"Hand {hand_id} Gesture: {gesture}",(10,30+30*hand_id),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        else:
            cv2.putText(frame,"No hand detected",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        return frame

# ----------------- Main ----------------- ## ----------------- Main application ----------------- #

def main():
    # Initialize tracker
    config = HandTrackerConfig(max_num_hands=2, smoothing_factor=0.3)
    tracker = HandTracker(config)

    # Start webcam
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Mirror image
            if config.mirror_image:
                frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            mp_result = tracker.landmarker.detect_for_video(rgb_frame, tracker._timestamp_ms())

            if mp_result.hand_landmarks:
                for hand_id, hand_landmarks in enumerate(mp_result.hand_landmarks):
                    # Draw landmarks
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        x1 = int(hand_landmarks[start_idx].x * frame.shape[1])
                        y1 = int(hand_landmarks[start_idx].y * frame.shape[0])
                        x2 = int(hand_landmarks[end_idx].x * frame.shape[1])
                        y2 = int(hand_landmarks[end_idx].y * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for lm in hand_landmarks:
                        cx = int(lm.x * frame.shape[1])
                        cy = int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # Update cursor
                    index_tip = hand_landmarks[8]  # Index finger tip
                    tracker._update_cursor(hand_id, index_tip.x, index_tip.y)

                    # Detect gesture
                    gesture = tracker.detect_gesture(hand_id)
                    tracker.hands[hand_id].gesture = gesture
                    cv2.putText(frame, f'Hand {hand_id}: {gesture}', (10, 30 + 30*hand_id),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            cv2.imshow("Dual-Hand Tracker", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()