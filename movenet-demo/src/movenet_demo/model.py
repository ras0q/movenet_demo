import tensorflow as tf
import tensorflow.python.trackable.autotrackable as autotrackable
from cv2.typing import MatLike


class MoveNet:
    module: autotrackable.AutoTrackable
    input_size: int

    def __init__(self, module: autotrackable.AutoTrackable, input_size: int):
        self.module = module
        self.input_size = input_size

    def predict(self, input_image: MatLike) -> tf.Tensor:
        model = self.module.signatures["serving_default"]
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores
