import os
import numpy as np
import sys
import json
import uuid  # For generating unique IDs
import random  # For generating random prices

# Ensure the app path is correctly added to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model.resnet50_model import extract_features  # Import your model

# Define dataset and output paths
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'ProjetDataset')  
images_dir = os.path.join('static', 'ProjetDataset')
output_dir = os.path.join(os.path.dirname(__file__), 'data')  # Output directory stays the same
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

embeddings_file = os.path.join(output_dir, 'embeddings.npy')
metadata_file = os.path.join(output_dir, 'metadata.json')

# Load existing embeddings and metadata if they exist
if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file).tolist()
else:
    embeddings = []

if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
else:
    metadata = []

# Process the dataset
image_paths = []
new_embeddings = []
new_metadata = []

print(f"Processing dataset: {dataset_dir}")
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path):  # Ensure it's a directory
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image_retrieval_path = os.path.join(images_dir, category, image_name)
            if os.path.isfile(image_path):  # Ensure it's a file
                if image_path in [item['image_path'] for item in metadata]:
                    print(f"Skipping already processed image: {image_path}")
                    continue
                try:
                    print(f"Processing: {image_path}")
                    embedding = extract_features(image_path)
                    new_embeddings.append(embedding.tolist())  # Convert to list for JSON serialization
                    
                    # Adding a unique ID and randomly generated price
                    image_id = str(uuid.uuid4())  # Generate a unique ID (UUID)
                    price = round(random.uniform(10.0, 500.0), 2)  # Random price between 10.0 and 500.0, rounded to 2 decimal places

                    new_metadata.append({
                        "id": image_id,
                        "image_path": image_retrieval_path,
                        "category": category,
                        "price": price
                    })
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

# Save updated embeddings
if new_embeddings:
    embeddings.extend(new_embeddings)  # Add new embeddings
    np.save(embeddings_file, np.array(embeddings))  # Save as .npy

# Save updated metadata
if new_metadata:
    metadata.extend(new_metadata)  # Add new metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

print(f"Processing complete. Updated embeddings and metadata saved!")
