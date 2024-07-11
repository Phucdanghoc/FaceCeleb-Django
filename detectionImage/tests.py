import numpy as np
from model.LSHModel import *
from model.FaceRecognitionModel import *
# Initialize the LSHModel
lsh_model = LSHModel()
weight_model = './model/restnet_v2.keras'
img_height = 224 
img_width = 224
num_classes = 1100
facerecognition_model = FaceRecognitionModel(weight_model,img_height,img_width,num_classes)
embedding_model = facerecognition_model.get_model()
# Load pre-trained ResNet101V2 weights
lsh_model.load_embedding_model(embedding_model)

# Load embeddings and labels from files
embeddings_file = './model/embedding/train_embeddings.npy'
labels_file = './model/embedding/train_labels.npy'
embeddings, labels = lsh_model.load_embeddings_and_labels(embeddings_file, labels_file)

# Create a DataFrame from the embeddings and labels
df = lsh_model.create_dataframe(embeddings, labels)

# Fit the LSH model
lsh_model.fit(df)

# Save the LSH model
lsh_model_path = 'lsh_model'
lsh_model.save_model(lsh_model_path)

# Save the training data
data_path = 'data'
lsh_model.save_data(data_path)