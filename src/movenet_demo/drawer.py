import cv2
from cv2.typing import MatLike


# 0: nose
# 1: left_eye # 2: right_eye
# 3: left_ear # 4: right_ear
# 5: left_shoulder # 6: right_shoulder
# 7: left_elbow # 8: right_elbow
# 9: left_wrist # 10: right_wrist
# 11: left_hip # 12: right_hip
# 13: left_knee # 14: right_knee
# 15: left_ankle # 16: right_ankle
def draw_joint_edges(
    frame: MatLike,
    keypoints_with_scores: list[list[float]],
    threshold: float = 0.2,
) -> MatLike:
    keypoint_edges = [
        (0, 1),  # nose - left_eye
        (1, 3),  # left_eye - left_ear
        (0, 2),  # nose - right_eye
        (2, 4),  # right_eye - right_ear
        (0, 5),  # nose - left_shoulder
        (5, 7),  # left_shoulder - left_elbow
        (7, 9),  # left_elbow - left_wrist
        (0, 6),  # nose - right_shoulder
        (6, 8),  # right_shoulder - right_elbow
        (8, 10),  # right_elbow - right_wrist
        (0, 11),  # nose - left_hip
        (11, 13),  # left_hip - left_knee
        (13, 15),  # left_knee - left_ankle
        (0, 12),  # nose - right_hip
        (12, 14),  # right_hip - right_knee
        (14, 16),  # right_knee - right_ankle
    ]

    image_height, image_width, _ = frame.shape
    for i, j in keypoint_edges:
        y_i, x_i, score_i = keypoints_with_scores[i]
        y_j, x_j, score_j = keypoints_with_scores[j]

        if score_i < threshold or score_j < threshold:
            continue

        cv2.line(
            frame,
            (int(x_i * image_width), int(y_i * image_height)),
            (int(x_j * image_width), int(y_j * image_height)),
            [0, 255, 0],
            2,
        )

    return frame


def draw_nose_keypoints(
    frame: MatLike,
    nose_keypoints_with_scores: list[list[float]],
    threshold: float = 0.2,
) -> MatLike:
    image_height, image_width, _ = frame.shape
    for y, x, score in nose_keypoints_with_scores:
        if score < threshold:
            continue

        cv2.circle(
            frame,
            (int(x * image_width), int(y * image_height)),
            5,
            [0, 0, 255],
            -1,
        )

    return frame


def draw_text(frame: MatLike, line: int, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (10, 30 * line),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        1,
    )
