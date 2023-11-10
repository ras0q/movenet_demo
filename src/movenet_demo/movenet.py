from enum import Enum
from typing import Any

import tensorflow as tf
import tensorflow_hub as tfhub
from cv2.typing import MatLike


class MoveNet:
    class ModelType(Enum):
        SINGLEPOSE_LIGHTNING = "singlepose-lightning"
        SINGLEPOSE_THUNDER = "singlepose-thunder"
        MULTIPOSE_LIGHTNING = "multipose-lightning"

        def load_model(self) -> Any:
            return tfhub.load(
                f"https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/{self.value.lower()}/versions/4"
            ).signatures["serving_default"]

        def input_size(self) -> int:
            if self == MoveNet.ModelType.SINGLEPOSE_LIGHTNING:
                return 192
            elif self == MoveNet.ModelType.SINGLEPOSE_THUNDER:
                return 256
            elif self == MoveNet.ModelType.MULTIPOSE_LIGHTNING:
                return 256
            else:
                raise ValueError(f"Invalid ModelType: {self}")

    model_type: "MoveNet.ModelType"
    model: Any

    def __init__(self, model_type: "MoveNet.ModelType"):
        self.model_type = model_type
        self.model = model_type.load_model()

    def predict(self, input_image: MatLike) -> tf.Tensor:
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = self.model(input_image)
        # NOTE: keypoints_with_scores consists of [y, x, score] for each keypoint.
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores
