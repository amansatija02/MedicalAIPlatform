import tensorflow as tf 

def get_data_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            # Randomly flip the image horizontally.
            tf.keras.layers.RandomFlip("horizontal"),
            
            # Rotate the image slightly to improve robustness.
            tf.keras.layers.RandomRotation(0.05),
            
            # Randomly zoom in or out within a small range.
            tf.keras.layers.RandomZoom(0.10),
            
            # Shift the image slightly in both directions.
            tf.keras.layers.RandomTranslation(height_factor= 0.05, width_factor= 0.05)    
        ],
        name = "data_augmentation",
    )