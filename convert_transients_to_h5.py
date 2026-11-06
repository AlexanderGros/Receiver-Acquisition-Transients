#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_transients_to_h5.py

Scans transient IQ capture directories, detects the signal start using power envelope
detection, checks steady-state tail convergence (EVM against nominal constellation),
extracts sliding 4096-sample windows, and saves them into an HDF5 dataset
alongside their metadata (modulation, alpha, sps, run index, window offset,
window index, tail EVM, and convergence flag).

Usage:
    python convert_transients_to_h5.py "C:\\Users\\Admin\\Desktop\\tests\\IQ_V5_trans" --output dataset.h5
"""

import os
import re
import argparse
import h5py
import numpy as np
from collections import defaultdict


# Normalized reference constellations for EVM computation
IDEAL_CONSTELLATIONS = {
    "BPSK": np.array([-1.0, 1.0], dtype=np.complex64),
    "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2.0),
    "8PSK": np.exp(1j * (np.arange(8) * (2.0 * np.pi / 8.0))),
    "QAM16": np.array([x + 1j * y for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]], dtype=np.complex64) / np.sqrt(10.0),
}

# Candidate phase ambiguity rotations for Costas loop locks
PHASE_ROTATIONS = {
    "BPSK": np.array([0.0, np.pi]),
    "QPSK": np.arange(4) * (np.pi / 2.0),
    "8PSK": np.arange(8) * (np.pi / 4.0),
    "QAM16": np.arange(4) * (np.pi / 2.0),
}


def verify_convergence(iq, modulation, tail_samples=40960, evm_threshold=0.38):
    """
    Evaluates whether the tail of the capture converged to the nominal constellation.
    Tests phase ambiguities and computes RMS EVM against ideal constellation points.
    Returns (is_converged: bool, best_evm: float).
    """
    if modulation not in IDEAL_CONSTELLATIONS:
        return True, 0.0

    if len(iq) < tail_samples:
        tail = iq
    else:
        tail = iq[-tail_samples:]

    power = np.mean(np.abs(tail) ** 2)
    if power < 1e-7:
        return False, 1.0

    norm_tail = tail / np.sqrt(power)
    constellation = IDEAL_CONSTELLATIONS[modulation]
    candidate_angles = PHASE_ROTATIONS.get(modulation, [0.0])

    best_evm = float("inf")

    for angle in candidate_angles:
        rotated_tail = norm_tail * np.exp(-1j * angle)
        # Compute min distance to ideal constellation for each sample
        dists = np.abs(rotated_tail[:, None] - constellation[None, :])
        min_dists = np.min(dists, axis=1)
        evm_rms = float(np.sqrt(np.mean(min_dists ** 2)))
        if evm_rms < best_evm:
            best_evm = evm_rms

    is_converged = best_evm <= evm_threshold
    return is_converged, best_evm


def load_iq(filename):
    iq = np.fromfile(filename, dtype=np.complex64)
    if len(iq) == 0:
        raise ValueError(f"The IQ file is empty: {filename}")
    return iq


def find_transient_start(iq, pre_margin=40960, threshold_factor=15.0, smooth_window=100):
    """
    Detects the rising edge of the signal power to locate the transient start.
    Goes back pre_margin samples to ensure noise before the burst is captured.
    """
    power = np.abs(iq) ** 2
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(power, kernel, mode='valid')
    
    noise_sample_count = min(1000, len(smoothed) // 10)
    if noise_sample_count < 10:
        return 0
        
    noise_floor = np.median(smoothed[:noise_sample_count])
    threshold = max(noise_floor * threshold_factor, 0.1)

    indices = np.where(smoothed > threshold)[0]
    if len(indices) == 0:
        return 0

    start_idx = max(0, indices[0] - pre_margin)
    return start_idx


def parse_metadata(filename):
    basename = os.path.basename(filename)
    
    # Example filename: QPSK_a0.400_sps2_corrected_run0.bin
    mod_match = re.match(r"^([^_]+)", basename)
    modulation = mod_match.group(1) if mod_match else "UNKNOWN"

    alpha_match = re.search(r"_a([^_]+)_", basename)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.4

    sps_match = re.search(r"_sps(\d+)_", basename)
    sps = int(sps_match.group(1)) if sps_match else 4

    run_match = re.search(r"run(\d+)", basename)
    run_idx = int(run_match.group(1)) if run_match else 0

    return modulation, alpha, sps, run_idx


def main():
    parser = argparse.ArgumentParser(description="Convert transient IQ captures into a 4096-sample HDF5 dataset using sliding windows.")
    parser.add_argument("input_dir", type=str, help="Path to root transient IQ directory")
    parser.add_argument("--output", type=str, default="transients_dataset.h5", help="Path to output HDF5 file")
    parser.add_argument("--window-size", type=int, default=4096, help="ML input size (default: 4096)")
    parser.add_argument("--hop-size", type=int, default=4096, help="Step size for sliding window (default: 4096, non-overlapping)")
    parser.add_argument("--max-samples-per-file", type=int, default=819200, help="Max samples to process after transient start per file (default: 819200)")
    parser.add_argument("--tail-samples", type=int, default=40960, help="Number of steady-state tail samples to evaluate for EVM (default: 40960)")
    parser.add_argument("--evm-threshold", type=float, default=0.38, help="RMS EVM threshold for convergence (default: 0.38)")
    parser.add_argument("--filter-unconverged", action="store_true", help="Discard captures that fail steady-state convergence check")
    parser.add_argument("--dry-run", action="store_true", help="Skip HDF5 generation and export metadata directly to CSV")
    parser.add_argument("--csv-output", type=str, default="transients_metadata.csv", help="Filename for the metadata CSV export in dry-run mode")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    iq_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bin"):
                iq_files.append(os.path.join(root, file))

    if not iq_files:
        print(f"No .bin files found in {args.input_dir}")
        return

    print(f"Found {len(iq_files)} .bin files. Processing and extracting non-overlapping windows...")

    all_windows = []
    all_modulations = []
    all_alphas = []
    all_sps = []
    all_runs = []
    all_offsets = []
    all_filenames = []
    all_window_indices = []
    all_is_converged = []
    all_tail_evm = []

    total_files_checked = 0
    unconverged_files_total = 0
    total_per_mod = defaultdict(int)
    unconverged_per_mod = defaultdict(int)

    for iq_file in iq_files:
        try:
            iq = load_iq(iq_file)
            modulation, alpha, sps, run_idx = parse_metadata(iq_file)

            total_files_checked += 1
            total_per_mod[modulation] += 1

            is_converged, tail_evm = verify_convergence(
                iq, modulation,
                tail_samples=args.tail_samples,
                evm_threshold=args.evm_threshold
            )

            if not is_converged:
                unconverged_files_total += 1
                unconverged_per_mod[modulation] += 1

            if args.filter_unconverged and not is_converged:
                print(f"[SKIP] {os.path.basename(iq_file)} did not converge (tail EVM: {tail_evm:.3f} > {args.evm_threshold})")
                continue

            start_idx = find_transient_start(iq, pre_margin=8192)
            
            # Extract up to max_samples_per_file after transient start (including pre-margin offset)
            end_idx = min(start_idx + args.max_samples_per_file, len(iq))
            segment = iq[start_idx:end_idx]

            # Slide window across the segment (non-overlapping when hop-size == window-size)
            window_idx = 0
            for pos in range(0, len(segment) - args.window_size + 1, args.hop_size):
                window = segment[pos : pos + args.window_size]
                if len(window) == args.window_size:
                    if not args.dry_run:
                        all_windows.append(window)
                    all_modulations.append(modulation)
                    all_alphas.append(alpha)
                    all_sps.append(sps)
                    all_runs.append(run_idx)
                    all_offsets.append(start_idx + pos)
                    all_filenames.append(os.path.basename(iq_file))
                    all_window_indices.append(window_idx)
                    all_is_converged.append(is_converged)
                    all_tail_evm.append(tail_evm)
                    window_idx += 1

        except Exception as e:
            print(f"Error processing {iq_file}: {e}")

    # Print convergence summary report
    print(f"\n--- CONVERGENCE SUMMARY ---")
    print(f"Total files checked: {total_files_checked}")
    print(f"Total unconverged files: {unconverged_files_total} ({(unconverged_files_total / max(1, total_files_checked)) * 100:.2f}%)")
    print("Unconverged files per modulation:")
    for mod, total in sorted(total_per_mod.items()):
        failed = unconverged_per_mod[mod]
        pct = (failed / max(1, total)) * 100.0
        print(f"  - {mod}: {failed}/{total} failed ({pct:.2f}%)")
    print(f"---------------------------\n")

    if not all_modulations:
        print("No windows extracted. Check your files and directory structure.")
        return

    if args.dry_run:
        print(f"Dry-run enabled. Exporting window metadata to CSV: {args.csv_output}...")
        import pandas as pd
        df_meta = pd.DataFrame({
            'filename': all_filenames,
            'modulation': all_modulations,
            'alpha': all_alphas,
            'sps': all_sps,
            'run_index': all_runs,
            'sample_offset': all_offsets,
            'window_index': all_window_indices,
            'is_converged': all_is_converged,
            'tail_evm': all_tail_evm
        })
        df_meta.to_csv(args.csv_output, index=False)
        print(f"CSV metadata export completed: {args.csv_output} ({len(df_meta)} windows)")
        return

    data_array = np.array(all_windows, dtype=np.complex64)
    
    print(f"Saving dataset to {args.output}...")
    print(f"Total windows extracted: {data_array.shape[0]}")
    print(f"Data shape: {data_array.shape}")

    # Encode string metadata for HDF5 compatibility
    dt_str = h5py.string_dtype(encoding='utf-8')

    with h5py.File(args.output, "w") as f:
        f.create_dataset("iq_data", data=data_array, compression="gzip")
        f.create_dataset("modulation", data=np.array(all_modulations, dtype=dt_str))
        f.create_dataset("alpha", data=np.array(all_alphas, dtype=np.float32))
        f.create_dataset("sps", data=np.array(all_sps, dtype=np.int32))
        f.create_dataset("run_index", data=np.array(all_runs, dtype=np.int32))
        f.create_dataset("sample_offset", data=np.array(all_offsets, dtype=np.int32))
        f.create_dataset("window_index", data=np.array(all_window_indices, dtype=np.int32))
        f.create_dataset("filename", data=np.array(all_filenames, dtype=dt_str))
        f.create_dataset("is_converged", data=np.array(all_is_converged, dtype=bool))
        f.create_dataset("tail_evm", data=np.array(all_tail_evm, dtype=np.float32))

    print("HDF5 dataset creation completed successfully!")


if __name__ == "__main__":
    main()
