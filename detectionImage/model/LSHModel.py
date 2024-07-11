import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from tensorflow.keras.preprocessing import image
from PIL import Image

class LSHModel:
    def __init__(self, app_name="BucketedRandomProjectionLSH", bucket_length=1.0, seed=12345):
        self.spark = SparkSession.builder.appName(app_name).getOrCreate()
        self.bucket_length = bucket_length
        self.seed = seed
        self.model = None
        self.label_to_index = None
        self.index_to_label = None
        self.df = None
    
    def load_embeddings_and_labels(self, embeddings_file, labels_file):
        embeddings = np.load(embeddings_file, allow_pickle=True)
        labels = np.load(labels_file, allow_pickle=True)
        return embeddings, labels
    
    def create_dataframe(self, embeddings, labels):
        # Create a dictionary to map original labels to index values
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Convert labels to index values
        indexed_labels = np.array([self.label_to_index[label] for label in labels])

        # Create a list of tuples (feature_vector, index_label)
        data_tuples = [(Vectors.dense(embedding.tolist()), int(index_label)) for embedding, index_label in zip(embeddings, indexed_labels)]

        # Create a PySpark DataFrame
        df = self.spark.createDataFrame(data_tuples, ["features", "index"])
        self.df = df
        return df
    
    def fit(self, df):
        brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=self.bucket_length, seed=self.seed)
        self.model = brp.fit(df)
        return self.model
    
    def transform(self, df):
        if self.model:
            return self.model.transform(df)
        else:
            raise ValueError("Model has not been trained yet.")
    
    def find_nearest_neighbors(self, df, sample_features, num_neighbors=5):
        if self.model:
            return self.model.approxNearestNeighbors(df, sample_features, numNearestNeighbors=num_neighbors)
        else:
            raise ValueError("Model has not been trained yet.")
    
    def display_labels(self, neighbors):
        predicted_labels = [self.index_to_label[row['index']] for row in neighbors.collect()]
        return predicted_labels
    
    def save_model(self, path):
        if self.model:
            self.model.write().overwrite().save(path)
        else:
            raise ValueError("Model has not been trained yet.")
    
    def load_model(self, path):
        self.model = BucketedRandomProjectionLSHModel.load(path)
    
    def save_data(self, path):
        if self.df:
            self.df.write.mode('overwrite').parquet(path)
        else:
            raise ValueError("Training data has not been set yet.")
    
    def load_data(self, path):
        self.df = self.spark.read.parquet(path)
    
    def load_embedding_model(self, model):
        self.embedding_model = model

    def find_nearest_for_image(self, image_path, num_neighbors=5):
        sample_features = self.extract_embedding(image_path)
        neighbors = self.find_nearest_neighbors(self.df, sample_features, num_neighbors)
        return neighbors

    def extract_embedding(self, image_path, target_size=(224, 224)):
        def preprocess_image(image_path, target_size):
            img = Image.open(image_path)
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Rescale pixel values to [0, 1]
            return img_array

        img_array = preprocess_image(image_path, target_size)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        embedding = self.embedding_model.predict(img_array)
        return Vectors.dense(embedding[0])
