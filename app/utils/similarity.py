# utils/similarity.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to find all similar images ranked by similarity
def find_all_similar_images(new_image_path, embeddings, image_paths, get_embedding_for_new_image):
    # Get the embedding for the new image using the preprocess function
    new_image_embedding = get_embedding_for_new_image(new_image_path)
    
    # Reshape the new image embedding to match the shape of precomputed embeddings
    new_image_embedding = np.array(new_image_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the new image and all precomputed embeddings
    similarities = cosine_similarity(new_image_embedding, embeddings)[0]
    
    # Sort all images by similarity score in descending order
    ranked_indices = np.argsort(similarities)[::-1]  # Sort by highest similarity first
    
    # Return a list of tuples (image_path, similarity_score)
    ranked_images = [(image_paths[i], similarities[i]) for i in ranked_indices]
    return ranked_images
