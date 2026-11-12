import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras import Model

from src.utils.exception import ModelException
from src.utils.logger import logger


def evaluate_model(model: Model, test_dataset, class_names: list, disease: str) -> dict:
    """Runs the model on the test data and returns the metrics main.py prints.

    The test dataset must not be shuffled, so the labels collected here
    are in the same order as the predictions."""

    try:
        logger.info("Evaluating %s model", disease)

        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
        probabilities = model.predict(test_dataset, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)

        if len(y_true) != len(y_pred):
            raise ModelException("Number of predictions does not match the test labels.")

        # a fixed label list keeps the matrix the same shape even if the model
        # never predicts one of the classes
        label_ids = list(range(len(class_names)))

        # weighted average because the classes are imbalanced,
        # zero_division=0 avoids warnings for a class that is never predicted
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=label_ids, average="weighted", zero_division=0
        )

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "classification_report": classification_report(
                y_true, y_pred, labels=label_ids, target_names=class_names, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids),
        }

        logger.info(
            "%s test accuracy: %.4f, weighted F1: %.4f",
            disease, results["accuracy"], results["f1_score"]
        )
        return results

    except ModelException:
        raise

    except Exception as e:
        logger.exception("Evaluation failed for %s", disease)
        raise ModelException(f"Unable to evaluate model for {disease}.") from e