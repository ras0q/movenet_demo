import argparse
import time

import cv2
import numpy as np
import tensorflow as tf
from movenet_demo import drawer
from movenet_demo.movenet import MoveNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="camera device index", type=int, default=0)
    parser.add_argument("--width", help="camera width", type=int, default=640)
    parser.add_argument("--height", help="camera height", type=int, default=480)
    parser.add_argument(
        "--model",
        help=f"model type ({', '.join([t.value for t in MoveNet.ModelType])})",
        type=MoveNet.ModelType,
        default=MoveNet.ModelType.SINGLEPOSE_THUNDER,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height
    model_type: MoveNet.ModelType = args.model

    print(
        "INFO: starting camera..."
        + f"\n\tdevice: {cap_device}"
        + f"\n\twidth: {cap_width}"
        + f"\n\theight: {cap_height}"
    )
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    print("INFO: loading model..." + f"\n\tmodel: {model_type.name}")
    movenet = MoveNet(model_type)

    qrcode_detector = cv2.QRCodeDetector()
    username = ""
    input_size = model_type.input_size()
    last_scored_sec = 0.0  # record results every 100ms to csv
    is_record_step = False
    start_tick_count = 0
    results: dict[float, tf.Tensor] = {}

    print("INFO: starting capture loop...")
    while True:
        ret, frame = cap.read()
        assert ret

        frame = cv2.flip(frame, 1)  # mirror frame
        input_image = cv2.resize(frame, dsize=(input_size, input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, input_size, input_size, 3)
        keypoints_with_scores = movenet.predict(input_image)
        keypoints_with_scores = np.squeeze(keypoints_with_scores)

        frame_drawed = drawer.draw_joint_edges(
            frame,
            keypoints_with_scores,
            threshold=0.2,
        )

        nose_keypoints_with_scores: list[list[float]] = []
        for v in results.values():
            nose_keypoints_with_scores.append(v[0])

        frame_drawed = drawer.draw_nose_keypoints(
            frame_drawed,
            nose_keypoints_with_scores,
            threshold=0.2,
        )

        if is_record_step:
            elapsed = (cv2.getTickCount() - start_tick_count) / cv2.getTickFrequency()
            drawer.draw_text(frame_drawed, 1, "recording...")
            drawer.draw_text(frame_drawed, 2, f"username: {username}")
            drawer.draw_text(frame_drawed, 3, f"elapsed: {elapsed:.3f} sec")
            drawer.draw_text(frame_drawed, 4, "exit: q")
            # record results every 100ms to csv
            if elapsed - last_scored_sec > 0.1:
                results[elapsed] = keypoints_with_scores
                last_scored_sec = elapsed
        else:
            drawer.draw_text(frame_drawed, 1, "press r to start recording")
            drawer.draw_text(frame_drawed, 2, f"username: {username}")
            if not username:
                drawer.draw_text(frame_drawed, 3, "set username from QR code")
                decoded, _, _ = qrcode_detector.detectAndDecode(frame_drawed)
                if decoded:
                    print(f"INFO: detected username from QR code: {decoded}")
                    username = decoded

        cv2.imshow("frame", frame_drawed)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("r") and not is_record_step:
            is_record_step = True
            start_tick_count = cv2.getTickCount()

    print("INFO: outputting results to csv...")
    csv_file = open(f"{__file__}/../data/results_{username}_{time.time()}.csv", "w")
    csv_file.write("sec," + ",".join([f"y_{i},x_{i},s_{i}" for i in range(17)]) + "\n")
    for i, result in results.items():
        csv_file.write(
            f"{i},"
            + ",".join([f"{y:.3f},{x:.3f},{s:.3f}" for (y, x, s) in result])
            + "\n"
        )

    print("INFO: quitting...")
    cap.release()
    cv2.destroyAllWindows()
