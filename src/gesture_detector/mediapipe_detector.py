"""
MediaPipe 单手检测模块
仅检测单手关键点，不含姿态
"""

import cv2
import mediapipe as mp
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """使用 MediaPipe 进行单手关键点检测"""

    def __init__(self, min_detection_confidence=0.55, min_tracking_confidence=0.55):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _pick_active_hand(self, hands_results):
        if not hands_results.multi_hand_landmarks:
            return None, "unknown"

        active_hand_landmarks = hands_results.multi_hand_landmarks[0]
        hand_label = "unknown"

        if hands_results.multi_handedness:
            hand_label = hands_results.multi_handedness[0].classification[0].label.lower()

        return active_hand_landmarks, hand_label

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        hands_results = self.hands.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        active_hand_landmarks, active_hand_label = self._pick_active_hand(hands_results)

        # 兼容字段：left/right 仅保留一个槽位
        if active_hand_label == "left":
            left_hand_landmarks = active_hand_landmarks
            right_hand_landmarks = None
        elif active_hand_label == "right":
            left_hand_landmarks = None
            right_hand_landmarks = active_hand_landmarks
        else:
            left_hand_landmarks = active_hand_landmarks
            right_hand_landmarks = None

        results = SimpleNamespace(
            pose_landmarks=None,  # 不再使用pose
            active_hand_landmarks=active_hand_landmarks,
            active_hand_label=active_hand_label,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
        )

        return image, results

    def draw_landmarks(self, image, results, use_styled=True):
        if use_styled:
            self._draw_styled_landmarks(image, results)
        else:
            self._draw_simple_landmarks(image, results)

    def _draw_simple_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.active_hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )

    def _draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image,
            results.active_hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

        label = getattr(results, "active_hand_label", "unknown").upper()
        cv2.putText(
            image,
            f"Hand: {label}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    def release(self):
        if self.hands:
            self.hands.close()

    def __del__(self):
        self.release()
