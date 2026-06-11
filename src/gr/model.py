import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings before import

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)

def build_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Project input into higher dimensional space
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.1)(x)

    # Multi-head attention to capture temporal dependencies
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization(epsilon=1e-6)(attention_output + x)

    # Compress temporal dimension into global representation
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)  # Classification head
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Output gesture class probabilities

    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    print("[This script is meant to be imported into train.py, not run standalone]")