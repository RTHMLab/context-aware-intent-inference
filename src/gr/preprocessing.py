from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences


def normalize_segments(segments):
    all_data = np.concatenate(segments, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0) + 1e-8
    normalized_segments = [(seg - mean) / std for seg in segments]
    return normalized_segments, mean, std


def preprocess_data(
    x_train_raw,
    y_train_raw,
    x_test_raw,
    y_test_raw,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Normalizing features...")
    x_train_norm, mean, std = normalize_segments(x_train_raw)
    x_test_norm = [(seg - mean) / std for seg in x_test_raw]

    norm_path = output_dir / "normalization_stats.npz"
    np.savez(norm_path, mean=mean, std=std)
    print(f"Saved normalization stats to {norm_path}")

    max_len = max(
        max(seg.shape[0] for seg in x_train_norm),
        max(seg.shape[0] for seg in x_test_norm),
    )
    num_features = x_train_norm[0].shape[1]

    print(f"Padding sequences to length: {max_len}")
    x_train = pad_sequences(
        x_train_norm,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )
    x_test = pad_sequences(
        x_test_norm,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    encoder_path = output_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved label encoder to {encoder_path}")

    return x_train, y_train, x_test, y_test, label_encoder, max_len, num_features
