# -*- coding: utf-8 -*-
# train_models.py

import os
import h5py
import numpy as np
import pickle
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from models import get_oshea_model, get_constellation_model, get_model_2

# Configuration parameters
EPOCHS = 50
BATCH_SIZE = 32
WEIGHTS_DIR = "weights"
NB = 128
TARGET_VEC_LEN = 4096

os.makedirs(WEIGHTS_DIR, exist_ok=True)

def check_gpu():
    """
    Check and print GPU availability.
    """
    gpus = tf.config.list_physical_devices('GPU')
    print("\n===============================")
    if gpus:
        print(f"GPU is available! Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU found. Training will run on CPU.")
    print("===============================\n")

def load_dataset(file_path):
    """
    Helper function to load dataset from an H5 file using correct keys,
    and crop waveforms to TARGET_VEC_LEN from the center.
    """
    with h5py.File(file_path, 'r') as f:
        X = f['waveforms'][:]  # shape: (32000, 2, 16384)
        Y = f['labels'][:]     # shape: (32000,)

    # Center crop to TARGET_VEC_LEN (4096) for all models
    current_len = X.shape[2]
    if current_len > TARGET_VEC_LEN:
        start = (current_len - TARGET_VEC_LEN) // 2
        end = start + TARGET_VEC_LEN
        X = X[:, :, start:end]

    return X, Y

def normalize_per_waveform(x):
    """
    Normalize every waveform independently to [-1,1].

    Input shape:
        (batch, 2, vec_len)

    Returns
        normalized array
    """
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)

    return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

def vd_mapper(data, n_bins=NB):
    vd = []
    for i in range(np.shape(data)[0]):
        H, _, _ = np.histogram2d(data[i][0], data[i][1], bins=n_bins)
        H = np.flip(H.T, 0)
        H = H.astype(np.float32)
        H = np.log1p(H)      # log(1 + count)
        H /= np.max(H) + 1e-12
        vd.append(H)
    vd = np.asarray(vd, dtype=np.float32)
    vd = np.expand_dims(vd, axis=-1)
    return vd

def train_and_save():
    parser = argparse.ArgumentParser(description="Train IQ and Constellation models.")
    parser.add_argument("--model", type=str, choices=["oshea", "constellation", "model2", "all"], default="all", help="Model to train")
    parser.add_argument("--dataset", type=str, choices=["corrected", "raw", "both"], default="corrected", help="Dataset to use")
    parser.add_argument("--channel", action="store_true", help="Use multi_data_channel directory instead of multi_data")
    args = parser.parse_args()

    check_gpu()

    #multidata_dir = "multi_data_channel" if args.channel else "multi_data"
    multidata_dir = "new_multi" if args.channel else "multi_data"

    datasets = {}
    if args.dataset in ["corrected", "both"]:
        datasets["corrected"] = os.path.join(multidata_dir, "corrected_dataset.h5")
    if args.dataset in ["raw", "both"]:
        datasets["raw"] = os.path.join(multidata_dir, "raw_dataset.h5")

    models_to_run = []
    if args.model in ["oshea", "all"]:
        models_to_run.append("oshea")
    if args.model in ["constellation", "all"]:
        models_to_run.append("constellation")
    if args.model in ["model2", "all"]:
        models_to_run.append("model2")

    for data_type, path in datasets.items():
        if not os.path.exists(path):
            print(f"Dataset not found at {path}. Skipping...")
            continue

        print(f"\n--- Loading {data_type} dataset from {path} ---")
        X, Y = load_dataset(path)

        # Normalize waveforms independently per waveform
        X = normalize_per_waveform(X)

        for model_name in models_to_run:
            print(f"\nTraining {model_name.upper()} model on {data_type} dataset...")

            if model_name == "constellation":
                X_data = vd_mapper(X, n_bins=NB)
                model = get_constellation_model(nb=NB)
            else:
                X_data = np.expand_dims(X, axis=-1)
                if model_name == "oshea":
                    model = get_oshea_model(vec_len=TARGET_VEC_LEN)
                elif model_name == "model2":
                    model = get_model_2(vec_len=TARGET_VEC_LEN)

            X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y, test_size=0.2, random_state=42)

            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                mode='max',
                restore_best_weights=True,
                verbose=1
            )

            history = model.fit(
                X_train, Y_train,
                validation_data=(X_test, Y_test),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[early_stopping],
                verbose=1
            )

            loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
            print(f"{model_name.capitalize()} ({data_type}) Test Accuracy: {accuracy:.5f}, Loss: {loss:.5f}")

            prefix = f"{model_name}_channel" if args.channel else model_name
            weights_path = os.path.join(WEIGHTS_DIR, f"{prefix}_{data_type}_weights.h5")
            model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")

            history_path = os.path.join(WEIGHTS_DIR, f"{prefix}_{data_type}_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(history.history, f)
            print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    train_and_save()
