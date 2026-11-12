import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from config.constants import EPOCHS
from src.training.model_compiler import compile_model
from src.training.callbacks import create_callbacks
from src.utils.exception import ModelException
from src.utils.logger import logger


def get_class_weights(y_train) -> dict:
    """Gives rare classes a higher weight. Pneumonia and DR data are very
    imbalanced, without this the model just predicts the biggest class."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)

    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_model(model: Model, train_dataset, validation_dataset, model_name: str, class_weight: dict = None):
    try:
        logger.info("Starting training for %s", model_name)

        model = compile_model(model)
        callbacks = create_callbacks(model_name)

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight
        )

        logger.info(
            "Training completed successfully for %s",
            model_name
        )

        return model, history

    except ModelException:
        raise

    except Exception as e:
        logger.exception("Training failed for %s", model_name)
        raise ModelException(
            f"Unable to train model: {model_name}"
        ) from e
