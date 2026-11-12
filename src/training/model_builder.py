from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

from config.constants import IMAGE_CHANNELS, DROPOUT_RATE, LAST_CONV_LAYER_NAME
from src.utils.exception import ModelException
from src.utils.logger import logger


def build_model(image_size: tuple, num_classes: int, weights: str | None = "imagenet") -> Model:
    """ResNet50 with a small classification head.

    The head is attached directly to the ResNet50 graph instead of putting ResNet50
    inside another model. This keeps the layer 'conv5_block3_out' reachable with
    model.get_layer(), which is what Grad-CAM needs.

    Inputs are expected to be already scaled with resnet50.preprocess_input."""

    try:
        if num_classes < 2:
            raise ModelException("A classification model needs at least 2 classes.")

        logger.info("Building ResNet50 model for %d classes", num_classes)

        base = ResNet50(
            include_top=False,
            weights=weights,
            input_shape=(image_size[0], image_size[1], IMAGE_CHANNELS)
        )

        # Only the new head is trained. A frozen BatchNorm also stays in inference
        # mode, so the small medical batches do not disturb the ImageNet statistics.
        base.trainable = False

        x = layers.GlobalAveragePooling2D(name="global_pool")(base.output)
        x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
        x = layers.Dense(128, activation="relu", name="head_dense")(x)
        x = layers.Dropout(DROPOUT_RATE, name="head_dropout_2")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = Model(inputs=base.input, outputs=outputs, name="medical_resnet50")

        # fail now instead of in the middle of the first Grad-CAM request
        model.get_layer(LAST_CONV_LAYER_NAME)

        return model

    except ModelException:
        raise

    except Exception as e:
        logger.exception("Failed to build the model")
        raise ModelException("Unable to build the model.") from e