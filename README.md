# 🧠 TensorFlow Unstructured Data Hands-On

This repository contains practical exercises focused on building deep learning models for unstructured data (text and images) using TensorFlow's native API. The exercises are designed to provide hands-on experience with common deep learning tasks without relying on high-level abstractions or external libraries.

## 📚 Exercises

### [Exercise 1: BBC Text Classification](./exercise-01/)
Build a text classifier using TensorFlow's native API to categorize news articles into different categories (e.g., tech, sport, politics). This exercise covers:
- Text preprocessing using `TextVectorization`
- Building an embedding-based model
- Text classification using dense layers
- Model training and evaluation

### [Exercise 2: Image Classification](./exercise-02/)
Implement an image classifier using TensorFlow's native API with a custom dataset. This exercise covers:
- Image data loading and preprocessing
- Data augmentation techniques
- Building a CNN architecture
- Training with multiple data splits (train/valid/test)

## 🎯 Learning Objectives

- Practice implementing deep learning models using TensorFlow's native API
- Understand the preprocessing pipeline for different types of unstructured data
- Gain hands-on experience with:
  - Text vectorization and embeddings
  - Convolutional Neural Networks (CNNs)
  - Data preprocessing and augmentation
  - Model training and evaluation
  - Working with real-world datasets

## 🛠️ Requirements

Each exercise has its own specific requirements, but generally, you'll need:
- Python 3.x
- TensorFlow
- NumPy
- Pandas (for text exercise)
- Matplotlib
- Seaborn
- ipykernel (for notebooks)

## 🚀 Getting Started

1. Clone this repository:
```bash
git clone https://github.com/CesarChalcoElias/hands-on-tf-unstructured-data.git
cd hands-on-tf-unstructured-data
```

2. Navigate to the exercise you want to work on:
```bash
cd exercise-01  # or exercise-02
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Follow the instructions in each exercise's README for specific setup and execution steps.

## 📁 Repository Structure

```
.
├── README.md
├── exercise-01/          # Text Classification
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   └── README.md
└── exercise-02/          # Image Classification
    ├── src/
    ├── notebooks/
    ├── data/
    └── README.md
```

## 📝 Notes

- Each exercise is self-contained with its own README, requirements, and configuration
- Sandbox notebooks are provided for exploration and testing
- The focus is on using TensorFlow's native API to better understand the underlying concepts

Happy coding! 🎉
