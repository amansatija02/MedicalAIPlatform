from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config.constants import MODELS_DIR, EARLY_STOPPING_PATIENCE, LR_REDUCTION_FACTOR, LR_REDUCTION_PATIENCE
from src.utils.exception import ModelException
from src.utils.logger import logger

def create_callbacks(model_name: str):
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.keras"
        early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
        reduce_learningrate = ReduceLROnPlateau(monitor="val_loss", factor = LR_REDUCTION_FACTOR, patience=LR_REDUCTION_PATIENCE, min_lr=1e-7)
        
        logger.info("Training callbacks created successfully.")
        
        return [early_stop, model_checkpoint, reduce_learningrate]
    
    except Exception as e:
        logger.exception("Failed to create training callbacks")
        raise ModelException("Unable to create training callback")