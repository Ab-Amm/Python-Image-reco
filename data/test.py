import os
import numpy as np
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

# Function to find the most similar image
def find_most_similar_image(new_image_path, embeddings, image_paths):
    # Get the embedding for the new image
    new_image_embedding = get_embedding_for_new_image(new_image_path)
    
    # Reshape the new image embedding to match the shape of precomputed embeddings
    new_image_embedding = np.array(new_image_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the new image and all precomputed embeddings
    similarities = cosine_similarity(new_image_embedding, embeddings)
    
    # Find the index of the most similar image
    most_similar_index = np.argmax(similarities)
    
    # Return the most similar image's path and similarity score
    return image_paths[most_similar_index], similarities[0][most_similar_index]

# Load the precomputed embeddings and image paths (from JSON)
embeddings = np.load('data/data/embeddings.npy')  # Load precomputed embeddings

# Load metadata (image paths and categories) from JSON file
with open('data/data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Extract image paths from the metadata
image_paths = [item['image_path'] for item in metadata]

# Test with a new image
new_image_path = 'data/test.jpg'  # Replace with the path to the new image you want to test

# Find the most similar image
most_similar_image, similarity_score = find_most_similar_image(new_image_path, embeddings, image_paths)

print(f"The most similar image is: {most_similar_image}")
print(f"Cosine similarity score: {similarity_score:.4f}")
