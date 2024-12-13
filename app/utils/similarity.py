from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the precomputed embeddings (from dataset images)
embeddings = np.load('data/embeddings.npy')  # Precomputed embeddings for the dataset
image_paths = open('data/image_paths.txt', 'r').readlines()  # Image paths for reference

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(embeddings)

def find_similar_images(new_embedding):
    """Find similar images from the dataset based on cosine similarity"""
    distances, indices = knn.kneighbors([new_embedding])
    similar_images = [{"image": image_paths[i].strip(), "distance": dist} for i, dist in zip(indices[0], distances[0])]
    return similar_images
