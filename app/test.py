# data/test.py
import os
import sys
import numpy as np
import json

# Add the project root directory to sys.path to make 'app' accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from utils
from utils.preprocess import get_embedding_for_new_image
from utils.similarity import find_most_similar_image

# Load the precomputed embeddings and image paths (from JSON)
embeddings = np.load('data/data/embeddings.npy')  # Load precomputed embeddings

# Load metadata (image paths and categories) from JSON file
with open('data/data/metadata.json', 'r') as f:
    metadata = json.load(f)
# Extract image paths from the metadata
image_paths = [item['image_path'] for item in metadata]

# Test with a new image
new_image_path = 'data/test.jpg'  # Replace with the path to the new image you want to test

# Find the most similar image using the similarity function
most_similar_image, similarity_score = find_most_similar_image(new_image_path, embeddings, image_paths, get_embedding_for_new_image)

print(f"The most similar image is: {most_similar_image}")
print(f"Cosine similarity score: {similarity_score:.4f}")
