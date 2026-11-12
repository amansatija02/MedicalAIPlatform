from tensorflow.keras import layers, Sequential


def get_data_augmentation() -> Sequential:
    """Light augmentation for scans. It is only used on training images
    (see create_dataset in main.py), never on validation or test data.

    The input here is still in the 0-255 range, ResNet50 scaling is applied after this."""

    # Strong changes can hide the very thing the model has to find
    # (a small tumor, a faint lesion), so every value is kept small.
    return Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation"
    )