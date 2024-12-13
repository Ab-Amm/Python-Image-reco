from flask import Flask, render_template, request, jsonify
from app.routes import init_routes

app = Flask(__name__)
init_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
