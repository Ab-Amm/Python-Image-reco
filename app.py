from flask import Flask
import os

# Determine the base directory (where app.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize the Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'app/templates'),  # Correct path for templates
    static_folder=os.path.join(BASE_DIR, 'app/static')       # Correct path for static files
)

# Folder to save uploaded images
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Import routes after the app has been initialized to avoid circular imports
from app.routes import configure_routes

# Configure routes
configure_routes(app)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
