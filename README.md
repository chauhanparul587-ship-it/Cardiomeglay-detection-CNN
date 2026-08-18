#  Cardiomegaly Detection from Chest X-Rays using CNN

A deep learning-based binary image classification system that analyzes chest X-ray images to classify them into two categories related to cardiomegaly.

Unlike traditional tabular ML approaches, this project uses a custom Convolutional Neural Network (CNN) to automatically learn spatial and visual patterns directly from X-ray images.

## 🔍 Problem Statement

Cardiomegaly refers to an enlarged heart, which can be an important indicator of underlying cardiovascular conditions.

The objective of this project is to build an image classification model capable of distinguishing between two classes of chest X-ray images.

## 🧠 Approach

This project uses a **supervised deep learning approach** based on a custom CNN architecture.

The model learns hierarchical visual features directly from chest X-ray images through multiple convolutional layers.

### CNN Architecture

- Conv2D layers for spatial feature extraction
- ReLU activation functions
- Batch Normalization for stable training
- MaxPooling2D for spatial downsampling
- Flatten layer for converting feature maps into vectors
- Dense layer with 256 neurons
- Dropout (0.5) for regularization
- Softmax output layer for binary classification

## 🏗️ Model Configuration

| Component | Configuration |
|---|---|
| Input Size | 224 × 224 |
| Classification Type | Binary Image Classification |
| CNN Framework | TensorFlow / Keras |
| Activation | ReLU + Softmax |
| Optimizer | Adamax |
| Learning Rate | 0.0005 |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Dropout | 0.5 |
| Output Classes | 2 |
| Total Parameters | ~52.27M |

## 🖼️ Image Preprocessing

Images are resized to **224 × 224 pixels** and normalized using:

`pixel_value / 255`

This scales pixel values to the `[0,1]` range before being passed to the network.

## 🔄 Data Augmentation

To improve generalization and reduce overfitting, training images are augmented using:

- Rotation: ±30°
- Width shift: 15%
- Height shift: 15%
- Zoom: 20%
- Horizontal flipping
- 20% validation split

These transformations allow the CNN to learn more robust visual representations rather than memorizing individual training images.

## 📊 Dataset

The dataset contains chest X-ray images belonging to **two classes**.

During preprocessing, the images are loaded using Keras `ImageDataGenerator` and organized into training, validation, and testing pipelines.

## ⚙️ Training

The model is trained using the Adamax optimizer with a learning rate of `0.0005`.

Training uses a separate validation generator to monitor model performance during training.

**Early Stopping** is also used to help prevent unnecessary training and overfitting.

## 📈 Evaluation

Model performance can be evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix
- Training/Validation Accuracy
- Training/Validation Loss

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PIL

## 🚀 Key Learning Outcomes

Through this project, I worked with:

- Convolutional Neural Networks
- Image classification
- Medical image preprocessing
- Data augmentation
- Batch normalization
- Dropout regularization
- CNN architecture design
- Model training and validation
- Classification evaluation
- TensorFlow/Keras model development
