from pathlib import Path 
import tensorflow as tf 
from config.config import settings
from config.constants import IMAGE_SIZE
from src.utils.exception import ConfigurationException
from src.utils.logger import logger

def load_dataset(dataset_path: Path, shuffle : bool = True) -> tf.data.Dataset:
    """ load image from dir and return a tensorflow dataset.
    Structure: 
        dataset/
            dis-1/
            dis-2/
            dis-3/"""
    if not dataset_path.exists():
        logger.error("Dataset directory not found: %s", dataset_path)
        raise ConfigurationException(
            f"Dataset directory doesn't exist: {dataset_path}"
        )
        
    if not dataset_path.is_dir():
        logger.error("%s is not a directory.", dataset_path)
        raise ConfigurationException(
            f"Expected a directory but found: {dataset_path}"
        )   
    
    logger.info("Logging dataset from %s", dataset_path)
    
    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            labels = "inferred",
            label_mode = "categorical",
            image_size = IMAGE_SIZE,
            batch_size = settings.batch_size,
            shuffle = shuffle,
            seed = settings.random_seed
        )
        logger.info("Dataset loaded Successfully")
        return dataset
    
    except Exception as e:
        logger.info("Unable to load dataset")
        raise ConfigurationException(
            f"Failed to load dataset from {dataset_path}"
        )

