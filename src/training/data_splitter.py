import numpy as np 
from sklearn.model_selection import train_test_split
from config.constants import TEST_SPLIT, VALIDATION_SPLIT, RANDOM_SEED
from src.utils.exception import ModelException
from src.utils.logger import logger 

def split_dataset(X: np.ndarray, y: np.ndarray):
    try:
        if(len(X) != len(y)):
            raise ModelException("Number of Samples donot match with number of labels")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y, shuffle=True)
        
        validation_ratio = VALIDATION_SPLIT/(1 - TEST_SPLIT)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=RANDOM_SEED, stratify=y_train, shuffle=True)
        
        logger.info("Dataset split completed successfully")
        logger.info("Training Samples: %d | Validation Samples: %d | Test Samples: %d", len(X_train), len(X_validation), len(X_test))
        
        return (X_train, X_validation, X_test, y_train, y_validation, y_test)

    except Exception as e:
        logger.exception("Failed to split dataset")
        raise ModelException("unable to split data") from e