import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.gr.data_loader import load_all_data, load_all_data_by_subject
from src.gr.model import build_transformer_model
from src.gr.preprocessing import preprocess_data


def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition model from Xsens trial windows.")
    parser.add_argument("--test_session", default=None, help="Optional held-out session for debugging")
    parser.add_argument("--target_subject", default="sub3", help="Held-out subject for subject-level base training")
    parser.add_argument("--joint_angles_root", default="data/extracted_JointAngles")
    parser.add_argument("--trials_root", default="data/trials")
    parser.add_argument("--output_dir", default="results/gr/models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--no_merge_sit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checking for GPU availability...")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
    else:
        print("No GPU detected. Training will use CPU.")

    print("Loading GR training windows...")

    if args.test_session:
        split_name = args.test_session
        print(f"Using session-level debug split: held-out session = {args.test_session}")

        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_all_data(
            test_session=args.test_session,
            joint_angles_root=args.joint_angles_root,
            trials_root=args.trials_root,
            merge_sit=not args.no_merge_sit,
        )
    else:
        split_name = args.target_subject.lower()
        print(f"Using subject-level split: held-out subject = {split_name}")

        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_all_data_by_subject(
            target_subject=split_name,
            joint_angles_root=args.joint_angles_root,
            trials_root=args.trials_root,
            merge_sit=not args.no_merge_sit,
        )

    print(f"Train segments: {len(x_train_raw)}")
    print(f"Test segments: {len(x_test_raw)}")

    if not x_train_raw or not x_test_raw:
        raise RuntimeError("No training or test data found.")

    print("Preprocessing data...")
    x_train, y_train, x_test, y_test, label_encoder, max_len, num_features = preprocess_data(
        x_train_raw,
        y_train_raw,
        x_test_raw,
        y_test_raw,
        output_dir=output_dir,
    )

    print("Classes:", list(label_encoder.classes_))
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    print("Building model...")
    model = build_transformer_model(
        input_shape=(max_len, num_features),
        num_classes=len(label_encoder.classes_),
    )

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("Training model...")
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    print("Evaluating on held-out data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    model_path = output_dir / f"transformer_gr_leaveout_{split_name}.keras"
    model.save(model_path)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
