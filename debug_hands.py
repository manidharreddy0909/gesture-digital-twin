"""
Diagnostic debug script to understand hand tracking behavior.

Run this to see what's happening with hand detection and cursor control.
"""

import time
import cv2
from config import CursorAIConfig
from hand_tracker import HandTracker, HandTrackerConfig

def debug_mode():
    """Run in debug mode with terminal output."""

    print("=" * 70)
    print("HAND TRACKING DEBUG MODE")
    print("=" * 70)

    cfg = CursorAIConfig(
        mirror_image=True,
        max_num_hands=2,
        enable_cursor_control=False,  # Don't move OS cursor
        primary_cursor_hand="left"
    )

    print(f"\nConfiguration:")
    print(f"  Mirror Image: {cfg.mirror_image}")
    print(f"  Max Hands: {cfg.max_num_hands}")
    print(f"  Primary Hand: {cfg.primary_cursor_hand}")
    print(f"\nExpected Behavior:")
    print(f"  When using LEFT hand → hand_id should be 0 → is_primary=TRUE")
    print(f"  When using RIGHT hand → hand_id should be 1 → is_primary=FALSE")
    print(f"  With BOTH hands → LEFT controls cursor")
    print(f"\n" + "=" * 70 + "\n")

    tracker_cfg = HandTrackerConfig(
        max_num_hands=cfg.max_num_hands,
        min_hand_detection_confidence=cfg.min_detection_confidence,
        min_hand_presence_confidence=cfg.min_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        mirror_image=cfg.mirror_image,
    )

    tracker = HandTracker(tracker_cfg)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        tracker.close()
        return

    print("Webcam opened. Press 'q' to quit.\n")

    frame_count = 0
    prev_hands_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            processed_frame, hands = tracker.process(frame)

            # Debug output every 10 frames or when hand count changes
            if frame_count % 10 == 0 or len(hands) != prev_hands_count:
                print(f"\n[Frame {frame_count}] Hands detected: {len(hands)}")

                for hand in hands:
                    # Check if this hand is primary
                    is_primary = (cfg.primary_cursor_hand.lower() == hand.handedness.lower())

                    print(f"  Hand {hand.hand_id}:")
                    print(f"    Handedness: {hand.handedness}")
                    print(f"    Is Primary (controls cursor): {is_primary}")
                    print(f"    Index Tip: ({hand.index_finger_tip[0]:.3f}, {hand.index_finger_tip[1]:.3f})")

                if len(hands) == 0:
                    print(f"  [No hands detected]")

                prev_hands_count = len(hands)

            # Add text overlay to frame
            cv2.putText(
                processed_frame,
                f"Hands: {len(hands)} | Primary: {cfg.primary_cursor_hand}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Draw hand IDs and primary status
            h, w = processed_frame.shape[:2]
            for hand in hands:
                x_px = int(hand.index_finger_tip[0] * w)
                y_px = int(hand.index_finger_tip[1] * h)

                is_primary = (cfg.primary_cursor_hand.lower() == hand.handedness.lower())
                color = (0, 255, 0) if is_primary else (0, 0, 255)  # Green if primary, red if not

                # Draw circle at fingertip
                cv2.circle(processed_frame, (x_px, y_px), 10, color, -1)

                # Draw text
                label = f"ID{hand.hand_id}: {hand.handedness}"
                if is_primary:
                    label += " [CURSOR]"
                cv2.putText(
                    processed_frame,
                    label,
                    (x_px + 15, y_px - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

            cv2.imshow("DEBUG: Hand Tracking", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("\n" + "=" * 70)
        print("Debug mode ended.")
        print("=" * 70)


if __name__ == "__main__":
    debug_mode()
