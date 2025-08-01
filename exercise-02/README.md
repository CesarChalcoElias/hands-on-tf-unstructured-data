# 🖼️ Exercise 2 — Image Classification (TensorFlow API)

## 🎯 Objective

Build an image classifier using the standard TensorFlow API layers to classify images from a custom dataset.

---

## 🧠 Instructions

### 1. Load the Dataset
- Use **TensorFlow** data utilities to load images from directories
- Check the dataset structure and visualize class distribution

📄 File: `src/dataloader.py`

---

### 2. Preprocess the Data
- Resize images to a standard size
- Normalize pixel values
- Apply data augmentation techniques

📄 File: `src/dataloader.py`

---

### 3. Build the Model (Functional API)
- Architecture:
  ```
  Input → Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Dropout → Dense(output)
  ```
- Use `categorical_crossentropy` as the loss function

📄 File: `src/model.py`

---

### 4. Train and Evaluate
- Split the dataset into training, validation and test sets
- Train the model and plot training/validation metrics
- Evaluate model performance on the test set

📄 File: `src/train.py`

---

### 5. Run the Project
- Create a clean script that connects all parts: data loading, model creation, training, and evaluation

📄 File: `main.py`

---

## 🏗️ Project Structure

```
exercise-02/
├── artifacts/
│   └── model_checkpoint.keras    # Saved model
├── data/
│   ├── train/                   # Training images
│   ├── valid/                   # Validation images
│   └── test/                    # Test images
├── notebooks/
│   └── sandbox.ipynb            # Exploration and testing
├── src/
│   ├── dataloader.py           # Data loading and preprocessing
│   ├── model.py                # Model architecture
│   └── train.py                # Training and evaluation logic
├── main.py                     # Entry-point script
└── README.md                   # Project documentation
```

---

## ✅ Requirements

The project requires the following dependencies:
- numpy
- tensorflow
- matplotlib
- seaborn
- ipykernel

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

The project uses the following parameters in the main script:
- `DATA_PATH`: Path to the dataset directories
- `IMAGE_SIZE`: Size of the input images
- `BATCH_SIZE`: Size of training batches
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for the optimizer
- `MODEL_PATH`: Path where the trained model will be saved

## 🚀 Usage

The main script can be run with default configuration values or with custom command-line arguments:

### Using Default Values
To run with the default values:
```bash
python main.py
```

### Using Command-line Arguments
You can override the training parameters using command-line arguments:

```bash
# Custom number of epochs
python main.py --epochs 50

# Custom batch size
python main.py --batch_size 32

# Custom learning rate
python main.py --learning_rate 0.001

# Multiple custom parameters
python main.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

Available arguments:
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Size of training batches (default: 32)
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--image_size`: Size of input images (default: 224)
- `--model_path`: Path to save the trained model (default: 'artifacts/model_checkpoint.keras')

## 📌 Notes

- The model uses standard TensorFlow layers without external libraries
- Images are automatically loaded from the data directory structure
- The model checkpoint is saved in the `artifacts/` directory
- A sandbox notebook is available for exploration in `notebooks/`

---

## 📂 Dataset Structure

The dataset is organized in the following structure:
```
data/
├── train/              # Training set
│   ├── class1/
│   ├── class2/
│   └── ...
├── valid/              # Validation set
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/               # Test set
    ├── class1/
    ├── class2/
    └── ...
```

---

Happy coding! 🎉
