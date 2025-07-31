import json
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from src.dataloader import prepare_data
from src.model import build_model


with open("config/config.json", "r") as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description="Train a text classification model.")
parser.add_argument("--epochs", type=int, default=config["EPOCHS"], help="Number of epochs for training.")
parser.add_argument("--batch_size", type=int, default=config["BATCH_SIZE"], help="Batch size for training.")

if __name__ == "__main__":

    print("Starting text classification model training pipeline...")
    args = parser.parse_args()

    # Data gathering and preparation
    print("Preparing data...")
    X,y = prepare_data(
        path=config["DATA_PATH"],
        target_name=config["TARGET_NAME"],
        text_field=config["TEXT_FIELD"],
        max_tokens=config["VOCAB_SIZE"],
        output_sequence_length=config["OUTPUT_LENGTH"],
        output_mode="int",
    )

    # Model building
    print("Building model...")
    model = build_model(
        num_classes=len(np.unique(y)),
        vocab_size=config["VOCAB_SIZE"],
        sequence_length=config["OUTPUT_LENGTH"]
    )

    # Data splitting
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(),
        y,
        test_size=0.2,
        random_state=config["SEED"]
    )

    # Model training
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Model evaluation
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Model Persistence
    print("Saving model...")
    model.save(config["MODEL_PATH"])
    print(f"Model saved to {config['MODEL_PATH']}")

    print("Pipeline completed successfully.")