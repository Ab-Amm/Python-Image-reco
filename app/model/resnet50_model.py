from tensorflow import ResNet50
from tensorflow  import preprocess_input
from tensorflow import load_img, img_to_array
import numpy as np

# Load ResNet50 model with ImageNet weights (exclude top layer)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    """Extract features from an image using ResNet50"""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()  # Return as a 1D array (embedding)
