import argparse
import copy
from typing import Sequence

import cv2
import numpy as np
import tensorflow_hub as hub
from movenet_demo.drawer import draw
from movenet_demo.model import MoveNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="camera device index", type=int, default=0)
    parser.add_argument("--width", help="camera width", type=int, default=640)
    parser.add_argument("--height", help="camera height", type=int, default=480)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    assert module is not None
    input_size = 192
    movenet = MoveNet(module, input_size)

    while True:
        ret, frame = cap.read()
        assert ret

        frame_copied = copy.deepcopy(frame)

        input_image = cv2.resize(frame, dsize=(input_size, input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, input_size, input_size, 3)
        keypoints_with_scores = movenet.predict(input_image)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        # keypoints_with_scores: [h, w, score]
        image_height, image_width, _ = frame.shape
        keypoints: list[Sequence[int]] = [
            [int(hw[1] * image_width), int(hw[0] * image_height)]
            for hw in keypoints_with_scores[:, :2]
        ]
        scores = keypoints_with_scores[:, 2]

        frame_drawed = draw(frame_copied, keypoints_with_scores, threshold=0.2)
        cv2.imshow("frame", frame_drawed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
