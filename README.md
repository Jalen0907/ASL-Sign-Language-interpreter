🖐️ Real-Time Sign Language Translator

A real-time American Sign Language (ASL) translation system that detects hand gestures using computer vision and translates them into readable text and spoken audio.

📌 Overview

This project uses computer vision and deep learning to recognize sign language gestures from a live camera feed and convert them into text and speech. The goal is to improve accessibility and communication for individuals who rely on sign language.

The system performs:

Real-time hand detection

Gesture classification using a Convolutional Neural Network (CNN)

Text output generation

Optional text-to-speech audio playback

🚀 Features

📷 Real-time webcam gesture detection

🧠 Custom CNN model for sign classification

🔤 Text output of recognized signs

🔊 Text-to-speech conversion

📊 Model evaluation with accuracy metrics

🧹 Preprocessing pipeline (resizing, normalization, landmark extraction)

🛠️ Tech Stack

Python

OpenCV – video capture & image processing

MediaPipe – hand landmark detection (if used)

TensorFlow / PyTorch – CNN model training

NumPy / Pandas – data processing

Matplotlib – visualization

pyttsx3 / gTTS – text-to-speech

🧠 Model Architecture

The gesture classification model is a Convolutional Neural Network consisting of:

Convolutional layers (feature extraction)

ReLU activation functions

Max pooling layers

Fully connected layers

Softmax output layer

Loss Function: Cross-Entropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Confusion Matrix
