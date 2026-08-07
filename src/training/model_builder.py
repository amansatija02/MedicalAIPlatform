from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import Model
from config.constants import IMAGE_CHANNELS, DROPOUT_RATE
from src.utils.exception import ModelException
from src.utils.logger import logger

def build_model(image_size: tuple[int, int], num_classes: int) -> Model:
    try:
        logger.info("Creating model")
        
        input_layer = Input(
            shape = (
                image_size[0], image_size[1], IMAGE_CHANNELS
            )
        ) 
        
        base_model = ResNet50(
            weights = "imagenet",
            include_top = False,
            input_tensor = input_layer
        )
        
        base_model.trainable = False 
        pooling_layer = GlobalAveragePooling2D()
        dropout_layer = Dropout(DROPOUT_RATE)
        classifier = Dense(units=num_classes, activation = "softmax")
        
        x = base_model.output 
        x = pooling_layer(x)
        x = dropout_layer(x)
        output_layer = classifier(x)
        
        model = Model(inputs = input_layer, outputs = output_layer, name = "MedicalAIModel")
        
        logger.info("Model Created Successfully")
        return model
    
    except Exception as e:
        logger.exception("Model Not Created")
        raise ModelException("Unable to create model")