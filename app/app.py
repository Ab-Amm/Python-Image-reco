# app.py
from flask import Flask
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Go up one level
from app.routes import configure_routes

# Initialize the Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'app/templates'))

# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure routes from routes.py
configure_routes(app)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
