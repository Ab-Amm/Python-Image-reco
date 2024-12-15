# data/test.py
import os
import sys
import numpy as np
import json

# Add the project root directory to sys.path to make 'app' accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from utils
from utils.preprocess import get_embedding_for_new_image
from utils.similarity import find_all_similar_images

# Load the precomputed embeddings and image paths (from JSON)
embeddings = np.load('../data/data/embeddings.npy')  # Load precomputed embeddings

# Load metadata (image paths and categories) from JSON file
with open('../data/data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Extract image paths from the metadata
image_paths = [item['image_path'] for item in metadata]

# Test with a new image
new_image_path = '../data/test2.jpg'  # Replace with the path to the new image you want to test

# Find all similar images using the similarity function
similar_images = find_all_similar_images(new_image_path, embeddings, image_paths, get_embedding_for_new_image)

# Print the top 5 most similar images with their similarity scores
print("Top 5 most similar images:")
for i, (image_path, similarity_score) in enumerate(similar_images[:5]):
    print(f"{i + 1}. {image_path} - Similarity Score: {similarity_score:.4f}")
