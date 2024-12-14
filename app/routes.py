from flask import request, render_template
import os
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Should point to the root of the project
from werkzeug.utils import secure_filename
from app.utils.preprocess import get_embedding_for_new_image
from app.utils.similarity import find_all_similar_images
import numpy as np
import json

def configure_routes(app):
    # Load precomputed embeddings and metadata
    embeddings = np.load('data/data/embeddings.npy')
    with open('data/data/metadata.json', 'r') as f:
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

        # Save the uploaded file to the 'uploads' folder
        upload_folder = '/uploads'
        os.makedirs(upload_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Process the uploaded image
        ranked_images = find_all_similar_images(
            file_path, embeddings, image_paths, get_embedding_for_new_image
        )

        # Render the result page with similar images
        return render_template('products.html', 
                               uploaded_image=file_path,
                               ranked_images=ranked_images)
