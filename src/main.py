import argparse
import copy
import time

import cv2
import numpy as np
from movenet_demo import drawer, model


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

    print(
        f"INFO: starting camera... (d: {cap_device}, w: {cap_width}, h: {cap_height})"
    )
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    print("INFO: loading model...")
    model_type = model.MoveNet.ModelType.SINGLEPOSE_THUNDER
    movenet = model.MoveNet(model_type)

    print("INFO: creating output csv file...")
    csv_file = open(f"{__file__}/../data/results_{time.time()}.csv", "w")
    csv_file.write("sec," + ",".join([f"y_{i},x_{i},s_{i}" for i in range(17)]) + "\n")

    print("INFO: starting capture loop...")
    input_size = model_type.input_size()
    threshold = 0.2
    last_scored_sec = 0.0  # record results every second to csv
    is_record_step = False
    start_tick_count = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        assert ret

        frame_copied = copy.deepcopy(frame)

        input_image = cv2.resize(frame, dsize=(input_size, input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, input_size, input_size, 3)
        keypoints_with_scores = movenet.predict(input_image)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        # record results every second to csv
        elapsed_sec = (cv2.getTickCount() - start_tick_count) / cv2.getTickFrequency()
        if elapsed_sec - last_scored_sec > 1.0:
            csv_file.write(
                f"{elapsed_sec:.3f},"
                + ",".join(
                    [
                        f"{yxs[0]:.3f},{yxs[1]:.3f},{yxs[2]:.3f}"
                        for yxs in keypoints_with_scores
                    ]
                )
                + "\n"
            )
            last_scored_sec = elapsed_sec

        frame_drawed = drawer.draw_joint_edges(
            frame_copied,
            keypoints_with_scores,
            threshold,
        )

        if is_record_step:
            drawer.draw_text(frame_drawed, 1, "recording...")
            drawer.draw_text(frame_drawed, 2, f"elapsed: {elapsed_sec:.3f} sec")
            drawer.draw_text(frame_drawed, 3, "exit: q")
        else:
            drawer.draw_text(frame_drawed, 1, "press r to start recording")

        cv2.imshow("frame", frame_drawed)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("r") and not is_record_step:
            is_record_step = True
            start_tick_count = cv2.getTickCount()

    cap.release()
    cv2.destroyAllWindows()
