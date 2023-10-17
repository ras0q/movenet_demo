import cv2
from cv2.typing import MatLike, Point


# 0: nose
# 1: left_eye # 2: right_eye
# 3: left_ear # 4: right_ear
# 5: left_shoulder # 6: right_shoulder
# 7: left_elbow # 8: right_elbow
# 9: left_wrist # 10: right_wrist
# 11: left_hip # 12: right_hip
# 13: left_knee # 14: right_knee
# 15: left_ankle # 16: right_ankle
def draw(
    frame: MatLike,
    keypoints: list[Point],
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

    for i in range(0, 17):
        for j in range(i, 17):
            if (i, j) in keypoint_edges:
                cv2.line(
                    frame,
                    keypoints[i],
                    keypoints[j],
                    [0, 255, 0],
                    2,
                )

    return frame
