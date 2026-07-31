from pathlib import Path
import cv2
import numpy as np 
import tensorflow as tf 
from src.models.model_loader import load_trained_model
from src.pipeline.preprocessing import preprocess_image
from src.utils.exception import ModelException
from src.utils.logger import logger 

def generate_gradcam(image_path: Path, disease: str, last_conv_layer_name: str) -> np.ndarray:
    """Generate Grad-CAM heatmap for prediction."""
    
    try:
        model = load_trained_model(disease)
        image = preprocess_image(image_path)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        image = np.expand_dims(image, axis = 0)
        
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
        heatmap /= tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        logger.info("Grad-CAM generated successfully.")
        return heatmap

    except Exception as e:
        logger.exception("Failed to generate Grad-CAM.")
        raise ModelException(
            "Unable to generate Grad - CAM"
        ) from e
    