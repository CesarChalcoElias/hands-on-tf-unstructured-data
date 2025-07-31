# 🔠 Exercise 1 — BBC Text Classification (TensorFlow API)

## 🎯 Objective

Build a text classifier that predicts the category of a news article (e.g., tech, sport, politics) using standard TensorFlow API layers.

---

## 🧠 Instructions

### 1. Load the Dataset
- Use **Pandas** to load `bbc-text.csv`
- Check column names and visualize class distribution

📄 File: `src/dataloader.py`

---

### 2. Preprocess the Data
- Use `TextVectorization()` to tokenize and pad text sequences
- Encode string labels using `LabelEncoder`

📄 File: `src/dataloader.py`

---

### 3. Build the Model (Functional API)
- Architecture:
  ```
  Input → Embedding → GlobalAveragePooling1D → Dense → Dropout → Dense(output)
  ```
- Use `sparse_categorical_crossentropy` as the loss function

📄 File: `src/model.py`

---

### 4. Train and Evaluate
- Split the dataset into training and test sets
- Train the model and plot training/validation loss curves
- Evaluate model performance on the test set

📄 File: `src/train.py`

---

### 5. Run the Project
- Create a clean script that connects all parts: data loading, model creation, training, and evaluation

📄 File: `main.py`

---

## 🏗️ Project Structure

```
exercise-01/
├── config/
│   └── config.json               # Configuration parameters
├── data/
│   └── bbc-text.csv             # Dataset file
├── models/
│   └── text_classification_model.keras  # Saved model
├── notebooks/
│   └── sandbox.ipynb            # Exploration and testing
├── src/
│   ├── __init__.py
│   ├── dataloader.py           # Data loading and preprocessing
│   ├── model.py                # Model architecture
│   └── train.py               # Training and evaluation logic
├── main.py                     # Entry-point script
└── README.md                   # Project documentation
```

---

## ✅ Requirements

The project requires the following dependencies:
- pandas
- numpy
- tensorflow
- scikit-learn
- matplotlib
- seaborn
- ipykernel

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

The project uses a configuration file (`config/config.json`) with the following parameters:
- `DATA_PATH`: Path to the dataset
- `TARGET_NAME`: Name of the target column (category)
- `TEXT_FIELD`: Name of the text column
- `VOCAB_SIZE`: Size of the vocabulary for text vectorization
- `OUTPUT_LENGTH`: Maximum sequence length
- `SEED`: Random seed for reproducibility
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Training batch size
- `MODEL_PATH`: Path where the trained model will be saved

## 🚀 Usage

The main script can be run with default configuration values or with custom command-line arguments:

### Using Default Values
To run with the default values from `config.json`:
```bash
python main.py
```

### Using Command-line Arguments
You can override the training parameters using command-line arguments:

```bash
# Custom number of epochs
python main.py --epochs 50

# Custom batch size
python main.py --batch_size 64

# Both custom epochs and batch size
python main.py --epochs 50 --batch_size 64
```

Available arguments:
- `--epochs`: Number of training epochs (default from config.json)
- `--batch_size`: Size of training batches (default from config.json)

## 📌 Notes

- The model uses standard TensorFlow layers without external libraries
- Configuration can be adjusted in `config/config.json`
- Trained models are saved in the `models/` directory
- A sandbox notebook is available for exploration in `notebooks/`

---

## 📂 Dataset Source

- [BBC-text dataset from Kaggle](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)

---

Happy coding! 🎉