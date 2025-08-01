from typing import Optional, Tuple
import tensorflow as tf

def train_model(
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    epochs: int = 10,
    callbacks: Optional[list] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    if callbacks is None:
        callbacks = []

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history


def evaluate_model(
        model: tf.keras.Model,
        test_data: tf.data.Dataset
) -> Tuple[float, float]:
    loss, accuracy = model.evaluate(test_data)
    return loss, accuracy



