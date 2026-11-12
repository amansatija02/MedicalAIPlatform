from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from config.constants import IMAGE_SIZE, LAST_CONV_LAYER_NAME
from src.models.model_loader import load_trained_model
from src.pipeline.preprocessing import preprocess_image, prepare_for_model
from src.utils.exception import MedicalAIException, ModelException
from src.utils.logger import logger

def generate_gradcam(image_path: Path, disease: str, last_conv_layer_name: str = LAST_CONV_LAYER_NAME) -> np.ndarray:
    """Generate Grad-CAM heatmap for prediction. Values are between 0 and 1."""

    try:
        model = load_trained_model(disease)
        image = preprocess_image(image_path)
        image = prepare_for_model(image)

        grad_model = tf.keras.models.Model(
            inputs = model.inputs,
            outputs= [
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ]
        )

        # Record the forward pass so gradients can be computed for Grad-CAM visualization.
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image)
            predicted_index = tf.argmax(predictions[0])
            loss = predictions[: , predicted_index]

        gradients = tape.gradient(loss, conv_output)

        # Average the gradients over the batch, height, and width dimensions.
        pooled_gradients = tf.reduce_mean(gradients, axis=(0,1,2)) # axis = 0 -> batch, axis = 1 -> height, axis = 2 -> width, axis = 3 -> Feature Map

        # Removing Batch Dimension as we are processing single image
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(pooled_gradients * conv_output, axis = -1)
        heatmap = tf.maximum(heatmap, 0)

        # If nothing activates, the max is 0 and a plain divide gives NaN everywhere
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, IMAGE_SIZE)

        logger.info("Grad-CAM generated successfully.")
        return heatmap

    except MedicalAIException:
        raise

    except Exception as e:
        logger.exception("Failed to generate Grad-CAM.")
        raise ModelException(
            "Unable to generate Grad - CAM"
        ) from e


def overlay_heatmap(image_path: Path, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Draws the heatmap on the image. Returns an RGB array that st.image can show."""
    try:
        image = preprocess_image(image_path)

        colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

    except MedicalAIException:
        raise

    except Exception as e:
        logger.exception("Failed to overlay heatmap.")
        raise ModelException("Unable to draw Grad-CAM heatmap.") from e
