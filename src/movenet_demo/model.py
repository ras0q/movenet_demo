from typing import Any

import tensorflow as tf
import tensorflow_hub as tfhub
from cv2.typing import MatLike


class MoveNet:
    module: Any
    input_size: int

    def __init__(self, moduleURL: str, input_size: int):
        self.module = tfhub.load(moduleURL)
        self.input_size = input_size

    def predict(self, input_image: MatLike) -> tf.Tensor:
        model = self.module.signatures["serving_default"]
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        # NOTE: keypoints_with_scores consists of [y, x, score] for each keypoint.
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores
