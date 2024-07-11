import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet101V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class FaceRecognitionModel:
    def __init__(self, weight_model, img_height, img_width, num_classes):
        """
        Initialize the FaceRecognitionModel.

        Parameters:
        weight_model (str): Path to the pre-trained weights.
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        num_classes (int): Number of classes for classification.
        """
        self.weight_model = weight_model
        self.img_height = img_height
        self.img_width = img_width        
        self.num_classes = num_classes
        
        self._build_model()
        self._load_weight_model()

    def _build_model(self):
        """Build the model architecture."""
        basemodel = ResNet101V2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )

        # Freeze the layers of the base model
        for layer in basemodel.layers:
            layer.trainable = False

        # Add custom layers on top of the base model
        top_layer = basemodel.output
        top_layer = Flatten()(top_layer)
        top_layer = Dropout(0.5)(top_layer)
        top_layer = Dense(units=1024, activation='relu')(top_layer)
        top_layer = Dropout(0.5)(top_layer)
        top_layer = Dense(self.num_classes, activation='sigmoid')(top_layer)

        # Create the final model
        self.model = Model(inputs=basemodel.input, outputs=top_layer)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )

    def _load_weight_model(self):
        """Load the pre-trained weights into the model."""
        self.model.load_weights(self.weight_model)

    def get_model(self):
        """Return the compiled model."""
        return self.model
