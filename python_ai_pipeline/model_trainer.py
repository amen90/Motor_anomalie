import os
import json
import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import build_dataset, ID_TO_CLASS_NAME


def standardize(X: np.ndarray) -> Tuple[np.ndarray, dict]:
    # Per-channel standardization
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = (X - mean) / std
    stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
    return Xn, stats


def build_cnn(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1]))  # (T, 3)
    x = tf.keras.layers.Conv1D(32, 9, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def to_tflite_int8(model: tf.keras.Model, rep_ds: np.ndarray, out_path: str):
    def rep_data_gen():
        for i in range(min(200, len(rep_ds))):
            x = rep_ds[i:i+1]
            yield [x.astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tflite)


def main():
    parser = argparse.ArgumentParser(description="Train motor anomaly CNN and export TFLite.")
    # Default dataset path: ../collected_data relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.normpath(os.path.join(script_dir, "..", "collected_data"))
    parser.add_argument("--data", default=default_data_dir, help="Path to collected_data directory")
    parser.add_argument("--window", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--step", type=float, default=0.5, help="Stride in seconds between windows")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    base_dir = args.data
    if not os.path.isdir(base_dir):
        # Also try project-root relative path if user ran from repo root
        alt_dir = os.path.normpath(os.path.join(script_dir, "..", base_dir))
        if os.path.isdir(alt_dir):
            base_dir = alt_dir
        else:
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    print(f"Using dataset directory: {base_dir}")

    X, y, sr, info = build_dataset(base_dir=base_dir,
                                   window_seconds=args.window,
                                   step_seconds=args.step)
    num_classes = int(len(set(y.tolist())))
    Xn, stats = standardize(X)

    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)

    model = build_cnn((X.shape[1], X.shape[2]), num_classes)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=args.epochs, batch_size=args.batch, callbacks=callbacks)

    # Evaluation
    test_probs = model.predict(X_test, batch_size=128)
    y_pred = np.argmax(test_probs, axis=1)
    print(classification_report(y_test, y_pred, target_names=[ID_TO_CLASS_NAME.get(i, str(i)) for i in sorted(set(y.tolist()))]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Export
    os.makedirs("models", exist_ok=True)
    model.save("models/motor_cnn_fp32.h5")
    with open("models/normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump({"mean": stats["mean"], "std": stats["std"], "info": info}, f, indent=2)
    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump(info["id_to_class"], f, indent=2)

    # INT8 TFLite
    to_tflite_int8(model, X_train[:500], "models/motor_cnn_int8.tflite")
    print("Saved models to ./models/")


if __name__ == "__main__":
    main()


