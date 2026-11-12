from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from config.constants import LEARNING_RATE
from src.utils.exception import ModelException
from src.utils.logger import logger


def compile_model(model: Model) -> Model:
    try:
        logger.info("Compiling CNN model")
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logger.info("Model Compiled Successfully")
        return model

    except Exception as e:
        logger.exception("Failed Compilation")
        raise ModelException("Unable to compile the model.") from e
