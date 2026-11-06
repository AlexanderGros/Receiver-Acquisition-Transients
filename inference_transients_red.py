# -*- coding: utf-8 -*-
# inference_transients_red.py

import os
import gc
import h5py
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from models import get_oshea_model, get_model_2, get_constellation_model

# The 4 modulations the RED models were trained on
possible_mods = ["bpsk", "qpsk", "8psk", "16qam"]

# Dictionary mapping model names and normalization types to their pre-trained weights paths
MODEL_WEIGHTS = {
    "oshea": {
        "minmax": "weights_4mods/out_oshea_MINMAX_4mods.keras",
        "power": "weights_4mods/out_oshea_POWER_4mods.keras",
    },
    "model_2": {
        "minmax": "weights_4mods/out_model_2_MINMAX_4mods.keras",
        "power": "weights_4mods/out_model_2_POWER_4mods.keras",
    },
    "constellation": {
        "minmax": "weights_4mods/out_constellation_MINMAX_4mods.keras",
        "power": "weights_4mods/out_constellation_POWER_4mods.keras",
    }
}

OUTPUT_DIR = "output_red_transients_4mods"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NB = 128  # constellation image size
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
        if 'modulation' in f:
            Y_str = f['modulation'][:]
        elif 'labels' in f:
            Y_str = f['labels'][:]
        else:
            raise KeyError(f"Could not find 'modulation' or 'labels' in H5 file. Available keys: {list(f.keys())}")
            
        window_indices = f['window_index'][:] if 'window_index' in f else np.zeros(len(Y_str), dtype=np.int32)
        filenames = f['filename'][:] if 'filename' in f else np.array([b"unknown"] * len(Y_str))

    mod_to_idx = {mod.lower(): i for i, mod in enumerate(possible_mods)}
    
    def decode_label(m):
        if isinstance(m, bytes):
            m = m.decode('utf-8')
        m_lower = m.strip().lower()
        if 'qam' in m_lower:
            m_lower = '16qam'  # Map all QAM variants to 16QAM
        return mod_to_idx.get(m_lower, -1)

    Y = np.array([decode_label(m) for m in Y_str])

    valid_idx = Y != -1
    print(f"Found {np.sum(valid_idx)} valid samples out of {len(Y)} total.")
    if not np.all(valid_idx):
        print(f"Warning: Filtering out {np.sum(~valid_idx)} samples with unknown modulation labels.")
        Y = Y[valid_idx]
        window_indices = window_indices[valid_idx]
        filenames = filenames[valid_idx]

    return Y, window_indices, filenames, valid_idx

def process_batch(X_complex_chunk, model_name, normalization="minmax"):
    """
    Converts complex chunk to 2-channel, normalizes per waveform (minmax or power), 
    and maps to constellation if needed.
    """
    # Convert complex64 to 2 channels (Real, Imag) -> shape: (N, 2, L)
    X = np.stack([X_complex_chunk.real, X_complex_chunk.imag], axis=1)

    # Center crop if needed
    current_len = X.shape[2]
    if current_len > TARGET_VEC_LEN:
        start = (current_len - TARGET_VEC_LEN) // 2
        end = start + TARGET_VEC_LEN
        X = X[:, :, start:end]
    elif current_len < TARGET_VEC_LEN:
        # Pad with zeros if shorter
        pad_total = TARGET_VEC_LEN - current_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        X = np.pad(X, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')

    # Normalization matching RED setup
    if normalization == "minmax":
        x_min = np.min(X, axis=(1, 2), keepdims=True)
        x_max = np.max(X, axis=(1, 2), keepdims=True)
        X = 2 * (X - x_min) / (x_max - x_min + 1e-8) - 1
    elif normalization == "power":
        # Power normalization (divide by RMS power)
        power = np.mean(np.square(X), axis=(1, 2), keepdims=True)
        X = X / np.sqrt(power + 1e-8)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if model_name == "constellation":
        vd = []
        for i in range(X.shape[0]):
            H, _, _ = np.histogram2d(X[i][0], X[i][1], bins=NB)
            H = np.flip(H.T, 0)
            H = H.astype(np.float32)
            H = np.log1p(H)
            max_val = np.max(H)
            if max_val > 0:
                H /= max_val + 1e-12
            vd.append(H)
        X_data = np.asarray(vd, dtype=np.float32)
        X_data = np.expand_dims(X_data, axis=-1)
    else:
        X_data = np.expand_dims(X, axis=-1)

    return X_data

def run_inference():
    parser = argparse.ArgumentParser(description="Run inference on transient HDF5 datasets using RED pre-trained models.")
    parser.add_argument("--model", type=str, choices=list(MODEL_WEIGHTS.keys()) + ["all"], default="all", help="Model to evaluate")
    parser.add_argument("--normalization", type=str, choices=["minmax", "power", "all"], default="all", help="Normalization type")
    parser.add_argument("--dataset", type=str, default="trans_data_channel_3.h5", help="Dataset filename")
    args = parser.parse_args()

    check_gpu()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found at {args.dataset}")

    Y_full, window_indices_full, filenames_full, valid_idx = load_metadata(args.dataset)

    models_to_run = list(MODEL_WEIGHTS.keys()) if args.model == "all" else [args.model]
    norms_to_run = ["minmax", "power"] if args.normalization == "all" else [args.normalization]

    with h5py.File(args.dataset, 'r') as f:
        if 'iq_data' in f:
            X_ds = f['iq_data']
        elif 'waveforms' in f:
            X_ds = f['waveforms']
        elif 'data' in f:
            X_ds = f['data']
        else:
            raise KeyError("Could not find dataset array ('iq_data', 'waveforms', or 'data') in H5 file.")

        total_samples = len(Y_full)

        for model_name in models_to_run:
            for norm in norms_to_run:
                weights_path = MODEL_WEIGHTS[model_name][norm]
                if not os.path.exists(weights_path):
                    print(f"Weights not found at {weights_path}, skipping.")
                    continue

                print(f"\n--- Running Inference for Model: {model_name}, Normalization: {norm} ---")
                
                # Load model architecture and weights
                if model_name == "oshea":
                    model = get_oshea_model(vec_len=TARGET_VEC_LEN, num_classes=len(possible_mods))
                elif model_name == "model_2":
                    model = get_model_2(vec_len=TARGET_VEC_LEN, num_classes=len(possible_mods))
                elif model_name == "constellation":
                    model = get_constellation_model(nb=NB, num_classes=len(possible_mods))
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                model.load_weights(weights_path)

                preds = []
                for start_idx in range(0, total_samples, BS):
                    end_idx = min(start_idx + BS, total_samples)
                    original_indices = np.where(valid_idx)[0][start_idx:end_idx]
                    
                    # Read batch from H5
                    chunk = X_ds[original_indices]
                    X_processed = process_batch(chunk, model_name, normalization=norm)
                    
                    batch_preds = model.predict(X_processed, verbose=0)
                    preds.append(batch_preds)

                preds = np.vstack(preds)
                y_pred = np.argmax(preds, axis=1)

                # Save results to CSV / dataframe
                df_results = pd.DataFrame({
                    'true_label': Y_full,
                    'pred_label': y_pred,
                    'window_index': window_indices_full,
                    'filename': [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn) for fn in filenames_full]
                })

                out_csv = Path(OUTPUT_DIR) / f"results_{model_name}_{norm}.csv"
                df_results.to_csv(out_csv, index=False)
                print(f"Saved results to {out_csv}")

                # Print classification report
                print(classification_report(Y_full, y_pred, target_names=possible_mods))

                tf.keras.backend.clear_session()
                gc.collect()

if __name__ == '__main__':
    run_inference()
