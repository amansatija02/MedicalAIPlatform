import numpy as np
from sklearn.model_selection import train_test_split

from config.constants import (
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    DATASETS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    PREDICTION_CLASSES,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    RANDOM_SEED
)

from src.utils.exception import ImageProcessingException
from src.utils.logger import logger
from src.pipeline.preprocessing import preprocess_image


def load_images(dataset_path, class_names):

    # First collect every file, so the array can be created once.
    # Images are kept as uint8 (0-255), float32 would need 4x the memory.
    files = []

    for label, class_name in enumerate(class_names):

        class_folder = dataset_path / class_name

        if not class_folder.exists():
            raise ImageProcessingException(
                f"Class folder not found: {class_folder}"
            )

        # sorted, so the split is the same on every machine
        class_files = [
            path for path in sorted(class_folder.iterdir())
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        if len(class_files) == 0:
            raise ImageProcessingException(
                f"No images found for class '{class_name}' in {class_folder}"
            )

        logger.info("Found %d images for class: %s", len(class_files), class_name)
        files.extend((path, label) for path in class_files)

    X = np.empty((len(files), IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS), dtype=np.uint8)
    y = np.empty(len(files), dtype=np.int32)

    loaded = 0
    skipped = 0

    for image_path, label in files:

        try:
            X[loaded] = preprocess_image(image_path)
            y[loaded] = label
            loaded += 1

        except Exception:
            skipped += 1
            logger.warning("Skipping corrupted image: %s", image_path)

    if skipped > 0:
        logger.warning("%d images were skipped in %s", skipped, dataset_path)

    if loaded == 0:
        raise ImageProcessingException(
            f"No valid images found in {dataset_path}"
        )

    return X[:loaded], y[:loaded]


def load_dataset(disease: str) -> tuple:
    """Returns training, validation and testing data with class names.
    Order: X_train, y_train, X_val, y_val, X_test, y_test, class_names"""

    if disease not in PREDICTION_CLASSES:
        raise ImageProcessingException(f"Unsupported disease: {disease}")

    dataset_path = DATASETS_DIR / disease

    training_path = dataset_path / "Training"
    testing_path = dataset_path / "Testing"

    if not training_path.exists() or not testing_path.exists():
        logger.error("Training or Testing folder not found: %s", dataset_path)
        raise ImageProcessingException(
            f"Training or Testing folder not found: {dataset_path}. "
            "Run download_datasets.py first."
        )

    try:

        class_names = PREDICTION_CLASSES[disease]

        # Load training images
        X_train, y_train = load_images(
            training_path,
            class_names
        )

        # Load testing images
        X_test, y_test = load_images(
            testing_path,
            class_names
        )

        # Split training data into training and validation
        validation_ratio = VALIDATION_SPLIT / (
            TRAIN_SPLIT + VALIDATION_SPLIT
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_ratio,
            random_state=RANDOM_SEED,
            stratify=y_train
        )

        # Show the class balance, the medical datasets are usually imbalanced
        logger.info(
            "%s class counts (train): %s",
            disease,
            dict(zip(class_names, np.bincount(y_train, minlength=len(class_names)).tolist()))
        )

        logger.info(
            "%s dataset loaded. Training: %d, Validation: %d, Testing: %d",
            disease,
            len(X_train),
            len(X_val),
            len(X_test)
        )

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            class_names
        )

    except ImageProcessingException:
        raise

    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise ImageProcessingException(
            f"Unable to load dataset for {disease}"
        ) from e
