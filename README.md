# AI-Face-Detection

# Project Overview

This project focuses on detecting and distinguishing between real and AI-generated faces. It involves loading and preprocessing image data, performing exploratory data analysis, extracting features using MobileNet, and training a neural network model for classification.

# Table of Contents

Installation

Data Collection

Exploratory Data Analysis

Image Preprocessing

Feature Extraction

Model Training

Results

Usage

Contributing

License

# Installation

To get started with this project, follow these steps:

Clone the repository

git clone https://github.com/IdajiliJohnOjochegbe/AI-Face-Detection.git

Navigate to the project directory:

cd aiface-detection

Install the required packages:

pip install -r requirements.txt

# Data Collection

Mount your Google Drive to access the datasets:


from google.colab import drive

drive.mount('/content/drive')

Verify the mounted drive:

import os

directory_path = '/content/drive/MyDrive/Data collection/Real images/thumbnails128x128'

if os.path.isdir(directory_path):
  
  files = os.listdir(directory_path)
 
  print(files)

else:

    print(f"{directory_path} is not a directory.")
    
Load and resize images:

def load_and_resize_images(directory, target_size=(128, 128)):

    # Code for loading and resizing images
    ...
Exploratory Data Analysis

Visualize sample images and analyze image sizes:

def visualize_samples(image_data, title, n=5):

    # Code for visualizing samples
    ...

def analyze_image_sizes(image_data, title):

    # Code for analyzing image sizes
    ...
# Image Preprocessing

Apply preprocessing techniques such as grayscale conversion and histogram equalization:


def preprocess_and_visualize(image_data, n=3):

    # Code for preprocessing and visualizing images
    ...
# Feature Extraction

Extract features using a pre-trained MobileNet model:


from tensorflow.keras.applications import MobileNet

# Load pre-trained MobileNet model

mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Function to extract features

def extract_features(image_data, model):

    # Code for feature extraction
    ...
# Model Training

Train a neural network model on the extracted features:

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

def create_nn_model(input_shape):

    # Code for creating the model
    ...

# Compile and train the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Results

Evaluate the model performance:

Evaluate the model

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')

print(f'Test Accuracy: {accuracy:.4f}')

# Usage

Provide usage instructions here. For example, how to preprocess new images and predict:

def preprocess_image(image_path, target_size=(128, 128)):

    # Code to preprocess new image
    ...

def predict_image(image_path, model):

    # Code to make predictions
    ...
# Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.

# License
This project is licensed under the MIT License.
