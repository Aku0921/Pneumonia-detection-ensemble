import os
import numpy as np
from tensorflow import keras
from PIL import Image
import io


class EnsemblePredictor:
    """Wrapper for DenseNet121 + VGG16 ensemble model."""
    
    def __init__(self, densenet_path, vgg16_path):
        """Load both models.
        
        Args:
            densenet_path: Path to DenseNet121 final_model.keras
            vgg16_path: Path to VGG16 final_model.keras
        """
        self.img_size = (224, 224)
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
        # Enable unsafe deserialization for Lambda layers
        keras.config.enable_unsafe_deserialization()
        
        try:
            self.densenet = keras.models.load_model(densenet_path)
            self.vgg16 = keras.models.load_model(vgg16_path)
            self.models_loaded = True
            print(f"✓ Loaded DenseNet121 from {densenet_path}")
            print(f"✓ Loaded VGG16 from {vgg16_path}")
        except Exception as e:
            self.models_loaded = False
            print(f"✗ Failed to load models: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for prediction.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image array of shape (1, 224, 224, 3) with values in [0, 255]
            Note: Models expect 0-255 range and handle normalization internally via Rescaling layer
        """
        if isinstance(image_path, str):
            # Load from file path
            img = Image.open(image_path).convert('RGB')
        else:
            # Assume it's a PIL Image object
            img = image_path.convert('RGB')
        
        # Resize to 224x224
        img = img.resize(self.img_size)
        
        # Convert to numpy array (values will be 0-255)
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path):
        """Make ensemble prediction on an image.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Dictionary with:
            - 'class': Predicted class ('NORMAL' or 'PNEUMONIA')
            - 'confidence': Confidence score (0-100)
            - 'ensemble_prob': Ensemble average probability
            - 'densenet_prob': DenseNet121 prediction
            - 'vgg16_prob': VGG16 prediction
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Cannot make predictions.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Get predictions from both models (disable verbose output)
        densenet_pred = self.densenet.predict(img_array, verbose=0)[0][0]
        vgg16_pred = self.vgg16.predict(img_array, verbose=0)[0][0]
        
        # Ensemble: average the two predictions
        ensemble_prob = (densenet_pred + vgg16_pred) / 2.0
        
        # Classify based on threshold of 0.5
        predicted_class = self.class_names[1] if ensemble_prob >= 0.5 else self.class_names[0]
        confidence = ensemble_prob if ensemble_prob >= 0.5 else (1.0 - ensemble_prob)
        confidence_percent = float(confidence * 100)
        
        return {
            'class': predicted_class,
            'confidence': round(confidence_percent, 2),
            'ensemble_prob': round(float(ensemble_prob), 4),
            'densenet_prob': round(float(densenet_pred), 4),
            'vgg16_prob': round(float(vgg16_pred), 4)
        }
