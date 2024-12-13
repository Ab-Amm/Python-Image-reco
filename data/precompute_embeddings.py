import os
import numpy as np
from app.model.resnet50_model import extract_features

dataset_dir = './data/ProjetDataset'
embeddings = []
image_paths = []

for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        embedding = extract_features(image_path)
        embeddings.append(embedding)
        image_paths.append(image_path)

np.save('data/embeddings.npy', np.array(embeddings))
with open('data/image_paths.txt', 'w') as f:
    f.writelines([path + '\n' for path in image_paths])
