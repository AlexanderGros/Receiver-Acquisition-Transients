# -*- coding: utf-8 -*-
# inference_transients.py

import os
import gc
import h5py
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from models import get_oshea_model, get_constellation_model, get_model_2, possible_mods, vec_len

WEIGHTS_DIR = "weights"
OUTPUT_DIR = "output_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NB = 128  # const image size
TARGET_VEC_LEN = 4096
BS = 32   # batch size

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("\n===============================")
    if gpus:
        print(f"GPU is available! Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU found. Running inference on CPU.")
    print("===============================\n")

def load_metadata(file_path):
    """
    Loads only metadata and labels from H5 file to keep system RAM usage minimal.
    """
    with h5py.File(file_path, 'r') as f:
        Y_str = f['modulation'][:]
        window_indices = f['window_index'][:] if 'window_index' in f else np.zeros(len(Y_str), dtype=np.int32)
        filenames = f['filename'][:] if 'filename' in f else np.array([b"unknown"] * len(Y_str))

    mod_to_idx = {mod: i for i, mod in enumerate(possible_mods)}
    Y = np.array([mod_to_idx.get(m.decode('utf-8') if isinstance(m, bytes) else m, -1) for m in Y_str])

    valid_idx = Y != -1
    if not np.all(valid_idx):
        print(f"Warning: Filtering out {np.sum(~valid_idx)} samples with unknown modulation labels.")
        Y = Y[valid_idx]
        window_indices = window_indices[valid_idx]
        filenames = filenames[valid_idx]

    return Y, window_indices, filenames, valid_idx

def process_batch(X_complex_chunk, model_name):
    """
    Converts complex chunk to 2-channel, normalizes per waveform, and maps to constellation if needed.
    """
    # Convert complex64 to 2 channels (Real, Imag) -> shape: (N, 2, 4096)
    X = np.stack([X_complex_chunk.real, X_complex_chunk.imag], axis=1)

    # Center crop if needed
    current_len = X.shape[2]
    if current_len > TARGET_VEC_LEN:
        start = (current_len - TARGET_VEC_LEN) // 2
        end = start + TARGET_VEC_LEN
        X = X[:, :, start:end]

    # Normalize waveforms independently per waveform to [-1, 1]
    x_min = np.min(X, axis=(1, 2), keepdims=True)
    x_max = np.max(X, axis=(1, 2), keepdims=True)
    X = 2 * (X - x_min) / (x_max - x_min + 1e-8) - 1

    if model_name == "constellation":
        vd = []
        for i in range(X.shape[0]):
            H, _, _ = np.histogram2d(X[i][0], X[i][1], bins=NB)
            H = np.flip(H.T, 0)
            H = H.astype(np.float32)
            H = np.log1p(H)
            H /= np.max(H) + 1e-12
            vd.append(H)
        X_data = np.asarray(vd, dtype=np.float32)
        X_data = np.expand_dims(X_data, axis=-1)
    else:
        X_data = np.expand_dims(X, axis=-1)

    return X_data

def run_inference():
    parser = argparse.ArgumentParser(description="Run inference on transient HDF5 datasets using trained models sequentially.")
    parser.add_argument("--model", type=str, choices=["oshea", "constellation", "model2", "all"], default="all", help="Model to evaluate")
    parser.add_argument("--dataset", type=str, default="trans_data_channel_3.h5", help="Path to transient HDF5 dataset")
    parser.add_argument("--weights-type", type=str, choices=["corrected", "raw"], default="corrected", help="Which trained weights prefix to use")
    parser.add_argument("--training-channel", action="store_true", help="Use weights trained with channel simulation")
    args = parser.parse_args()

    check_gpu()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found at {args.dataset}")

    print(f"\n--- Loading metadata from {args.dataset} ---")
    Y_full, window_indices_full, filenames_full, valid_idx = load_metadata(args.dataset)
    total_samples = len(Y_full)
    print(f"Loaded metadata for {total_samples} samples.")

    models_to_run = []
    if args.model in ["oshea", "all"]:
        models_to_run.append("oshea")
    if args.model in ["constellation", "all"]:
        models_to_run.append("constellation")
    if args.model in ["model2", "all"]:
        models_to_run.append("model2")

    for model_name in models_to_run:
        if args.training_channel:
            weights_filename = f"{model_name}_channel_{args.weights_type}_weights.h5"
        else:
            weights_filename = f"{model_name}_{args.weights_type}_weights.h5"
            
        weights_path = os.path.join(WEIGHTS_DIR, weights_filename)

        if not os.path.exists(weights_path):
            print(f"Weights not found at {weights_path}. Skipping model {model_name}...")
            continue

        print(f"\nRunning inference for {model_name.upper()} using weights from {weights_path}...")

        if model_name == "constellation":
            model = get_constellation_model(nb=NB)
        elif model_name == "oshea":
            model = get_oshea_model(vec_len=TARGET_VEC_LEN)
        elif model_name == "model2":
            model = get_model_2(vec_len=TARGET_VEC_LEN)

        model.load_weights(weights_path)

        # Stream predictions in chunks to avoid high system RAM usage
        all_preds = []
        chunk_size = 2048

        with h5py.File(args.dataset, 'r') as f:
            iq_dataset = f['iq_data']
            
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                
                # Find original indices considering valid_idx filter
                original_indices = np.where(valid_idx)[0][start_idx:end_idx]
                
                # Read chunk from HDF5
                X_complex_chunk = iq_dataset[original_indices]
                
                # Process chunk (normalize, optional constellation mapping)
                X_batch = process_batch(X_complex_chunk, model_name)
                
                # Predict batch
                preds_chunk = model.predict(X_batch, batch_size=BS, verbose=0)
                all_preds.append(preds_chunk)
                
                print(f"Processed samples {start_idx} to {end_idx} / {total_samples}", end='\r')

        print()
        preds = np.concatenate(all_preds, axis=0)
        pred_labels = np.argmax(preds, axis=1)

        # Evaluate overall loss and accuracy using a generator-like approach or direct metrics
        # Since sparse categorical crossentropy can be computed:
        correct_preds = (pred_labels == Y_full)
        accuracy = np.mean(correct_preds)
        
        # Approximate loss using sparse categorical crossentropy formula or evaluating per batch if needed
        # For simplicity and speed, compute cross entropy loss directly:
        eps = 1e-15
        preds_clipped = np.clip(preds, eps, 1 - eps)
        ce_loss = -np.mean(np.log(preds_clipped[np.arange(len(Y_full)), Y_full]))

        print(f"{model_name.capitalize()} Transient Test Accuracy: {accuracy:.5f}, Loss: {ce_loss:.5f}")

        print("\nClassification Report:")
        print(classification_report(Y_full, pred_labels, target_names=possible_mods))

        print("Confusion Matrix:")
        print(confusion_matrix(Y_full, pred_labels))

        # Save results to CSV for sliding window accuracy plotting
        results_df = pd.DataFrame({
            'filename': [f.decode('utf-8') if isinstance(f, bytes) else f for f in filenames_full],
            'window_index': window_indices_full,
            'true_label_idx': Y_full,
            'true_modulation': [possible_mods[i] for i in Y_full],
            'pred_label_idx': pred_labels,
            'pred_modulation': [possible_mods[i] for i in pred_labels],
            'correct': correct_preds,
            'model': model_name,
            'weights_type': args.weights_type
        })

        if args.training_channel:
            csv_filename = f"results_{model_name}_channel_{args.weights_type}.csv"
        else:
            csv_filename = f"results_{model_name}_{args.weights_type}.csv"

        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        results_df.to_csv(csv_path, index=False)
        print(f"Saved detailed results to {csv_path}")

        # Clear VRAM between models
        del model
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    run_inference()
