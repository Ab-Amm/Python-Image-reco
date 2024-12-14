# utils/similarity.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to find the most similar image
def find_most_similar_image(new_image_path, embeddings, image_paths, preprocess_func):
    # Get the embedding for the new image using the preprocess function
    new_image_embedding = preprocess_func(new_image_path)
    
    # Reshape the new image embedding to match the shape of precomputed embeddings
    new_image_embedding = np.array(new_image_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the new image and all precomputed embeddings
    similarities = cosine_similarity(new_image_embedding, embeddings)
    
    # Find the index of the most similar image
    most_similar_index = np.argmax(similarities)
    
    # Return the most similar image's path and similarity score
    return image_paths[most_similar_index], similarities[0][most_similar_index]
