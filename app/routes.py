from flask import request, render_template
import os
import numpy as np
import json
from werkzeug.utils import secure_filename
from app.utils.preprocess import get_embedding_for_new_image
from app.utils.similarity import find_all_similar_images

import re

def sanitize_filename(filename):
    # Remove special characters and replace spaces with underscores
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename

def configure_routes(app):
    # Load precomputed embeddings and metadata
    embeddings_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data', 'embeddings.npy')
    embeddings = np.load(embeddings_path)
    
    metadata_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data', 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    image_paths = [item['image_path'] for item in metadata]

    # Route: Display the upload form
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    # Route: Handle image upload and return similar images
    @app.route('/upload', methods=['POST'])
    def upload_image():
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No file selected", 400

        # Assuming the uploaded image is saved in a folder within static/uploads
        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Process the uploaded image
        ranked_images = find_all_similar_images(
            file_path, embeddings, image_paths, get_embedding_for_new_image
        )

        # Prepare data for the frontend
        similar_images_data = []

        print("Ranked Images (before modification):", ranked_images)

        # Debugging Relative Paths and Metadata
        for image_tuple in ranked_images:
            image_path = image_tuple[0]  # Extract the path from the tuple
            print("Processing Image:", image_path)

            if 'ProjetDataset' in image_path:
                relative_path = image_path.replace('\\', '/')
                
                # Find the metadata for the image based on its path
                image_metadata = next((item for item in metadata if item['image_path'] == image_path), None)
                if image_metadata:
                    similar_images_data.append({
                        'id': image_metadata['id'],
                        'image_path': relative_path,
                        'category': image_metadata['category'],
                        'price': image_metadata['price']
                    })

        # Debugging: Print final data structure
        print("Similar Images Data:", similar_images_data)

        # Render the template with updated relative paths and additional metadata
        return render_template(
            'products.html',
            ranked_images=similar_images_data
        )
