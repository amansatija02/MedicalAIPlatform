import gc
import sys
import numpy as np
import tensorflow as tf
from config.constants import (
    SUPPORTED_DISEASES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)

from src.pipeline.augmentation import get_data_augmentation
from src.training.dataset_loader import load_dataset
from src.training.model_builder import build_model
from src.training.trainer import train_model, get_class_weights
from src.training.savemodel import save_model
from src.evaluation.evaluator import evaluate_model

from src.utils.exception import ModelException
from src.utils.logger import logger


def create_dataset(X, y, training=False):
    # The images stay in the numpy array and are handed over one at a time.
    # from_tensor_slices() would copy every split again, and shuffle(len(X))
    # would keep a second full copy of the training images in memory.
    rng = np.random.default_rng(RANDOM_SEED)

    def generator():
        # Images are stored class by class, so the training order is shuffled
        # and a new order is used in every epoch.
        order = rng.permutation(len(X)) if training else np.arange(len(X))
        for i in order:
            yield X[i], y[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.batch(BATCH_SIZE)

    augmentation = get_data_augmentation()

    def prepare(images, labels):
        images = tf.cast(images, tf.float32)

        # augmentation only on training data, never on validation or test
        if training:
            images = augmentation(images, training=True)

        # same scaling that prepare_for_model() uses at prediction time
        images = tf.keras.applications.resnet50.preprocess_input(images)
        return images, labels

    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_disease(disease):
    logger.info("Starting pipeline for %s", disease)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        class_names
    ) = load_dataset(disease)

    train_dataset = create_dataset(X_train, y_train, training=True)
    validation_dataset = create_dataset(X_validation, y_validation)
    test_dataset = create_dataset(X_test, y_test)

    class_weights = get_class_weights(y_train)
    logger.info("Class weights for %s: %s", disease, class_weights)

    model = build_model(IMAGE_SIZE, len(class_names))

    model, history = train_model(
        model,
        train_dataset,
        validation_dataset,
        disease,
        class_weights
    )

    results = evaluate_model(
        model,
        test_dataset,
        class_names,
        disease
    )

    model_path = save_model(model, disease)

    logger.info(
        "Pipeline completed for %s. Model saved at %s",
        disease,
        model_path
    )

    print("\nEvaluation Results:", disease)
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1_score"])
    print(results["classification_report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


def run_training(diseases):
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    failed = []

    for disease in diseases:
        try:
            train_disease(disease)

        except Exception:
            # one missing dataset should not stop the other models
            logger.exception("Training failed for %s", disease)
            failed.append(disease)

        finally:
            # clear the old graph and image arrays before the next disease
            tf.keras.backend.clear_session()
            gc.collect()

    if failed:
        raise ModelException(
            f"Training failed for: {', '.join(failed)}"
        )

    logger.info("All disease models trained successfully.")


if __name__ == "__main__":
    # python main.py                  -> trains every disease
    # python main.py "Pneumonia"      -> trains only the given disease(s)
    selected = sys.argv[1:] or list(SUPPORTED_DISEASES)

    for name in selected:
        if name not in SUPPORTED_DISEASES:
            raise SystemExit(
                f"Unknown disease '{name}'. Choose from: {', '.join(SUPPORTED_DISEASES)}"
            )

    run_training(selected)