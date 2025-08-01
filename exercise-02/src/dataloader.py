from typing import Tuple
import numpy as np
import tensorflow as tf

def load_image_dataset(
    data_dir: str = "data",
    batch_size: int = 32,
    img_size: tuple = (224, 224)
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size
    )

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        image_size=img_size,
        batch_size=batch_size
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size
    )

    return train_ds, valid_ds, test_ds

def get_preprocessing_pipeline():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1)
    ])