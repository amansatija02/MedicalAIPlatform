from pathlib import Path
import shutil
import re
from dotenv import load_dotenv

# load .env before kagglehub is used, so KAGGLE_API_TOKEN is available
load_dotenv()

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import (
    DATASETS_DIR,
    PREDICTION_CLASSES,
    RANDOM_SEED
)

DOWNLOAD_DIR = DATASETS_DIR / "_downloads"

# Each disease is first built here and only moved into data/datasets when it is
# complete, so a failed run never leaves a half filled disease folder behind.
STAGING_DIR = DATASETS_DIR / "_staging"

BRAIN_TUMOR_ID = "masoudnickparvar/brain-tumor-mri-dataset"
PNEUMONIA_ID = "paultimothymooney/chest-xray-pneumonia"
APTOS_COMPETITION_ID = "aptos2019-blindness-detection"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_name(name):
    """Normalize folder names for matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_prepared(disease):
    """True when the disease folder already contains images."""
    target = DATASETS_DIR / disease

    if not target.exists():
        return False

    return any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in target.rglob("*")
    )


def download_brain_tumor():
    try:
        return Path(
            kagglehub.dataset_download(
                BRAIN_TUMOR_ID,
                output_dir=str(DOWNLOAD_DIR / "Brain_Tumor")
            )
        )
    except Exception as e:
        raise RuntimeError(
            "Brain Tumor download failed. Check your internet connection and "
            "Kaggle login (KAGGLE_USERNAME / KAGGLE_KEY or kaggle.json)."
        ) from e


def download_pneumonia():
    try:
        return Path(
            kagglehub.dataset_download(
                PNEUMONIA_ID,
                output_dir=str(DOWNLOAD_DIR / "Pneumonia")
            )
        )
    except Exception as e:
        raise RuntimeError(
            "Pneumonia download failed. Check your internet connection and "
            "Kaggle login (KAGGLE_USERNAME / KAGGLE_KEY or kaggle.json)."
        ) from e


def download_aptos():
    try:
        return Path(
            kagglehub.competition_download(
                APTOS_COMPETITION_ID,
                output_dir=str(DOWNLOAD_DIR / "APTOS")
            )
        )
    except Exception as e:
        # most common reason: the competition rules were never accepted
        raise RuntimeError(
            "APTOS download failed. Open the 'APTOS 2019 Blindness Detection' "
            "competition on kaggle.com and accept its rules with the same account "
            "that you use for the API key, then run this again."
        ) from e


def find_class_split(root, split_name, class_names):
    """Find a split folder containing all expected class folders."""
    expected = {normalize_name(name) for name in class_names}

    for folder in root.rglob("*"):
        if not folder.is_dir() or folder.name.lower() != split_name.lower():
            continue

        found = {
            normalize_name(item.name)
            for item in folder.iterdir()
            if item.is_dir()
        }

        if expected.issubset(found):
            return folder

    raise FileNotFoundError(
        f"Could not find '{split_name}' folder with classes {class_names} in {root}"
    )


def copy_class_images(source, destination, class_names):
    """Copy class images while standardizing destination folder names."""
    destination.mkdir(parents=True, exist_ok=True)

    source_folders = {
        normalize_name(folder.name): folder
        for folder in source.iterdir()
        if folder.is_dir()
    }

    for class_name in class_names:
        key = normalize_name(class_name)

        if key not in source_folders:
            raise FileNotFoundError(
                f"Class '{class_name}' not found in {source}"
            )

        source_class = source_folders[key]
        target_class = destination / class_name
        target_class.mkdir(parents=True, exist_ok=True)

        for image_path in source_class.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # zip files made on a Mac contain "._name.jpg" files that are not images
            if image_path.name.startswith("._"):
                continue

            target_file = target_class / image_path.name

            # two source folders can have the same file name (for example train and val),
            # keep both instead of silently overwriting
            counter = 1
            while target_file.exists():
                target_file = target_class / f"{image_path.stem}_{counter}{image_path.suffix}"
                counter += 1

            shutil.copy2(image_path, target_file)

        if not any(target_class.iterdir()):
            raise FileNotFoundError(
                f"No images were copied for class '{class_name}' from {source_class}"
            )


def finish_disease(disease):
    """Moves the finished staging folder into data/datasets."""
    staged = STAGING_DIR / disease
    target = DATASETS_DIR / disease

    if target.exists():
        # only folders without any file (like the empty skeleton) can be replaced
        if any(item.is_file() for item in target.rglob("*")):
            raise FileExistsError(
                f"{target} has files but no images. Move or remove this folder and run again."
            )

        shutil.rmtree(target)

    shutil.move(str(staged), str(target))


def prepare_brain_tumor(source):
    disease = "Brain Tumor"
    classes = PREDICTION_CLASSES[disease]

    training = find_class_split(source, "Training", classes)
    testing = find_class_split(source, "Testing", classes)

    staged = STAGING_DIR / disease

    copy_class_images(training, staged / "Training", classes)
    copy_class_images(testing, staged / "Testing", classes)

    finish_disease(disease)
    print("Brain Tumor dataset prepared.")


def prepare_pneumonia(source):
    disease = "Pneumonia"
    classes = PREDICTION_CLASSES[disease]

    training = find_class_split(source, "train", classes)
    testing = find_class_split(source, "test", classes)

    staged = STAGING_DIR / disease

    copy_class_images(training, staged / "Training", classes)
    copy_class_images(testing, staged / "Testing", classes)

    # The original validation folder is small. Add it to Training so that
    # the project loader can create a fresh validation split from Training.
    validation = None

    try:
        validation = find_class_split(source, "val", classes)
    except FileNotFoundError:
        pass

    if validation is not None:
        copy_class_images(validation, staged / "Training", classes)

    finish_disease(disease)
    print("Pneumonia dataset prepared.")


def prepare_diabetic_retinopathy(source):
    disease = "Diabetic Retinopathy"

    csv_files = list(source.rglob("train.csv"))
    image_folders = [
        folder for folder in source.rglob("train_images")
        if folder.is_dir()
    ]

    if not csv_files or not image_folders:
        raise FileNotFoundError(
            "APTOS train.csv or train_images folder was not found."
        )

    data = pd.read_csv(csv_files[0])

    required_columns = {"id_code", "diagnosis"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "APTOS train.csv must contain id_code and diagnosis columns."
        )

    # Build the id -> file lookup once. Searching the folder again for every row
    # (3600+ rows) made this step extremely slow.
    image_lookup = {
        path.stem: path
        for path in image_folders[0].rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    classes = PREDICTION_CLASSES[disease]

    # APTOS competition test images do not have public labels.
    # Split the labelled training data into project Training and Testing.
    training_data, testing_data = train_test_split(
        data,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=data["diagnosis"]
    )

    staged = STAGING_DIR / disease

    for split_name, split_data in [
        ("Training", training_data),
        ("Testing", testing_data)
    ]:
        for class_id, class_name in enumerate(classes):
            class_folder = staged / split_name / class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            selected_rows = split_data[
                split_data["diagnosis"] == class_id
            ]

            for image_id in selected_rows["id_code"].astype(str):
                source_image = image_lookup.get(image_id)

                if source_image is None:
                    raise FileNotFoundError(f"APTOS image not found: {image_id}")

                shutil.copy2(
                    source_image,
                    class_folder / source_image.name
                )

    finish_disease(disease)
    print("Diabetic Retinopathy dataset prepared.")


DOWNLOADERS = {
    "Brain Tumor": download_brain_tumor,
    "Pneumonia": download_pneumonia,
    "Diabetic Retinopathy": download_aptos,
}

PREPARERS = {
    "Brain Tumor": prepare_brain_tumor,
    "Pneumonia": prepare_pneumonia,
    "Diabetic Retinopathy": prepare_diabetic_retinopathy,
}


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Only the missing diseases are downloaded. Earlier the script stopped with
    # an error if even one disease was already there.
    pending = []

    for disease in PREDICTION_CLASSES:
        if is_prepared(disease):
            print(f"{disease} already prepared, skipping. "
                  f"Delete {DATASETS_DIR / disease} to download it again.")
        else:
            pending.append(disease)

    try:
        for disease in pending:
            print(f"\nDownloading {disease} dataset...")
            source = DOWNLOADERS[disease]()

            PREPARERS[disease](source)

            # remove this source right away, APTOS alone is several GB
            shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    finally:
        # also clean up when something failed in the middle
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        shutil.rmtree(STAGING_DIR, ignore_errors=True)

    print("\nAll datasets are ready.")
    print("Dataset location:", DATASETS_DIR)


if __name__ == "__main__":
    main()
