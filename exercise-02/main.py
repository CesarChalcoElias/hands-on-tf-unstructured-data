import argparse
import tensorflow as tf
from src.dataloader import load_image_dataset, get_preprocessing_pipeline
from src.model import get_model
from src.train import train_model, evaluate_model

argparser = argparse.ArgumentParser(description="Train a CNN model on image data.")
argparser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset.')
argparser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
argparser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
argparser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
argparser.add_argument('--image_size', type=int, default=224, help='Size of input images.')
argparser.add_argument('--model_path', type=str, default='artifacts/model_checkpoint.keras', help='Path to save the trained model.')

if __name__ == "__main__":
    args = argparser.parse_args()
    
    print("Loading datasets...")
    train_data, val_data, test_data = load_image_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.image_size, args.image_size)
    )

    print("Creating preprocessing pipeline...")
    preprocessing_pipeline = get_preprocessing_pipeline()

    print("Applying preprocessing to datasets...")
    train_data = train_data.map(lambda x, y: (preprocessing_pipeline(x), y))
    val_data = val_data.map(lambda x, y: (preprocessing_pipeline(x), y))
    test_data = test_data.map(lambda x, y: (preprocessing_pipeline(x), y))

    print("Building model...")
    model = get_model()

    print("Training model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.model_path,
        monitor='val_loss',
        save_best_only=True
    )
    
    # Configure optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    model, history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        callbacks=[early_stopping, checkpoint]
    )
        
    print("Evaluating model...")
    loss, accuracy = evaluate_model(model=model, test_data=test_data)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    print(f"Training complete. Model saved to '{args.model_path}'")
    print("Training pipeline ended successfully.")