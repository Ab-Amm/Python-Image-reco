# utils/preprocess.py
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from app.model.resnet50_model import extract_features

# Function to extract embedding for the new image
def get_embedding_for_new_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features using the same model used for precomputing embeddings
    return extract_features(image_path)