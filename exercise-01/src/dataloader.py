from typing import Tuple

import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization

from sklearn.preprocessing import LabelEncoder


def load_raw_data(path: str):
    return pd.read_csv(path)

def vectorize_text(data: pd.DataFrame, text_field: str, **kwargs) -> Tuple[TextVectorization, np.array]:
    vectorizer = TextVectorization(**kwargs)
    vectorizer.adapt(data[text_field].to_numpy())
    return vectorizer, vectorizer(data[text_field].to_numpy())

def encode_labels(target_name: str, data: pd.DataFrame) -> Tuple[LabelEncoder, np.array]:
    encoder = LabelEncoder()
    encoder.fit(data[target_name])
    return encoder, encoder.transform(data[target_name])

def prepare_data(path: str, target_name: str, text_field: str, **kwargs) -> Tuple[np.array, np.array]:
    raw_data = load_raw_data(path)
    vectorizer, vectorized_text = vectorize_text(raw_data, text_field, **kwargs)
    encoder, encoded_labels = encode_labels(target_name, raw_data)
    return vectorized_text, encoded_labels

