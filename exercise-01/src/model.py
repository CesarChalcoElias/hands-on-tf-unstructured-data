from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout


def build_model(
    num_classes: int,
    vocab_size: int = 1000,
    sequence_length: int = 100
):
    inputs = Input(shape=(sequence_length,))

    x = Embedding(input_dim=vocab_size, output_dim=64)(inputs)
    x = GlobalAveragePooling1D()(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model