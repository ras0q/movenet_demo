import argparse

import cv2
import numpy as np
import tensorflow_hub as hub
from movenet_demo.model import MoveNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)

    return parser.parse_args()


# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}


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
    frame: np.ndarray,
    keypoints: np.ndarray,
) -> np.ndarray:
    for i in range(0, 17):
        for j in range(i, 17):
            if (i, j) in KEYPOINT_EDGE_INDS_TO_COLOR:
                cv2.line(
                    frame,
                    keypoints[i],
                    keypoints[j],
                    KEYPOINT_EDGE_INDS_TO_COLOR[(i, j)],
                    2,
                )

    return frame


if __name__ == "__main__":
    args = get_args()
    cap_device: int = args.device

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    assert module is not None
    input_size = 192
    movenet = MoveNet(module, input_size)

    while True:
        ret, frame = cap.read()
        assert ret

        input_image = cv2.resize(frame, dsize=(input_size, input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, input_size, input_size, 3)
        keypoints_with_scores = movenet.predict(frame)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        # keypoints_with_scores: tuple of (y, x, score)
        keypoints = keypoints_with_scores[:, :2].map(
            lambda yx: (yx[1] * 640, yx[0] * 480)
        )
        scores = keypoints_with_scores[:, 2]

        drawed = draw(frame, keypoints)

        cv2.imshow("frame", drawed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
