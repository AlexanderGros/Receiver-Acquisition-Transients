#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_h5_convergence.py

Checks HDF5 recordings convergence from concatenated tail windows or direct waveforms datasets,
outputs periodic summaries (overall and per-modulation counts), and saves batch constellation plots.
"""

import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Normalized reference constellations matching the transmission scripts
IDEAL_CONSTELLATIONS = {
    "BPSK": np.array([-1.0, 1.0], dtype=np.complex64),
    "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2.0),
    "8PSK": np.exp(1j * (np.arange(8) * (2.0 * np.pi / 8.0))),
    "QAM16": np.array([x + 1j * y for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]], dtype=np.complex64) / np.sqrt(10.0),
}

PHASE_ROTATIONS = {
    "BPSK": np.array([0.0, np.pi]),
    "QPSK": np.arange(4) * (np.pi / 2.0),
    "8PSK": np.arange(8) * (np.pi / 4.0),
    "QAM16": np.arange(4) * (np.pi / 2.0),
}

MODULATION_MAP = {
    "BPSK": 0,
    "QPSK": 1,
    "8PSK": 2,
    "QAM16": 3
}


def verify_convergence(iq, modulation, evm_threshold=0.38):
    if modulation not in IDEAL_CONSTELLATIONS:
        return True, 0.0

    power = np.mean(np.abs(iq) ** 2)
    if power < 1e-7:
        return False, float("inf")

    norm_iq = iq / np.sqrt(power)
    constellation = IDEAL_CONSTELLATIONS[modulation]
    candidate_angles = PHASE_ROTATIONS.get(modulation, [0.0])

    best_evm = float("inf")
    for angle in candidate_angles:
        rotated = norm_iq * np.exp(-1j * angle)
        dists = np.abs(rotated[:, None] - constellation[None, :])
        min_dists = np.min(dists, axis=1)
        evm_rms = float(np.sqrt(np.mean(min_dists ** 2)))
        if evm_rms < best_evm:
            best_evm = evm_rms

    return best_evm <= evm_threshold, best_evm


def parse_modulation_from_filename(filename):
    basename = os.path.basename(filename)
    for mod in MODULATION_MAP.keys():
        if basename.startswith(mod):
            return mod
    return "UNKNOWN"


def save_constellation_plot(iq, mod, title, output_path):
    plt.figure(figsize=(5, 5))
    plt.scatter(iq.real, iq.imag, s=1, alpha=0.5, c='blue')
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    plt.title(f"{mod} - {title}")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def process_file(file_path, evm_threshold, samples_per_mod=200, output_dir="convergence_plots"):
    total = 0
    converged = 0
    mod_stats = defaultdict(lambda: {"converged": 0, "not_converged": 0})
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        with h5py.File(file_path, 'r') as f:
            # Check for ML dataset format: waveforms (N, 2, L), modulation_names
            if 'waveforms' in f:
                waveforms = f['waveforms']
                num_waveforms = waveforms.shape[0]
                
                modulation_names_ds = f['modulation_names'][:] if 'modulation_names' in f else None
                labels = f['labels'][:] if 'labels' in f else None

                # Helper to extract modulation string for a given index i
                def get_modulation(i):
                    if modulation_names_ds is not None and len(modulation_names_ds) == num_waveforms:
                        m = modulation_names_ds[i]
                        if isinstance(m, bytes):
                            return m.decode('utf-8')
                        return str(m)
                    elif labels is not None:
                        label_val = labels[i]
                        if isinstance(label_val, (bytes, str, np.bytes_, np.str_)):
                            return label_val.decode('utf-8') if isinstance(label_val, bytes) else str(label_val)
                        elif modulation_names_ds is not None and 0 <= int(label_val) < len(modulation_names_ds):
                            m = modulation_names_ds[int(label_val)]
                            if isinstance(m, bytes):
                                return m.decode('utf-8')
                            return str(m)
                        else:
                            return str(label_val)
                    elif modulation_names_ds is not None and len(modulation_names_ds) > 0:
                        m = modulation_names_ds[0]
                        if isinstance(m, bytes):
                            return m.decode('utf-8')
                        return str(m)
                    return "UNKNOWN"

                # Group waveform indices by modulation
                mod_to_indices = defaultdict(list)
                for i in range(num_waveforms):
                    mod = get_modulation(i)
                    mod_to_indices[mod].append(i)

                indices = []
                for mod, idxs in mod_to_indices.items():
                    n_take = min(len(idxs), samples_per_mod) if samples_per_mod > 0 else len(idxs)
                    chosen = np.random.choice(idxs, n_take, replace=False)
                    indices.extend(chosen)
                np.random.shuffle(indices)

                selected_count = len(indices)
                batch_mod_samples = defaultdict(lambda: {"converged": [], "not_converged": []})

                for idx, i in enumerate(indices):
                    row = waveforms[i]  # shape (2, L)
                    iq = row[0] + 1j * row[1]
                    mod = get_modulation(i)

                    is_conv, evm = verify_convergence(iq, mod, evm_threshold=evm_threshold)
                    if is_conv:
                        converged += 1
                        mod_stats[mod]["converged"] += 1
                        if len(batch_mod_samples[mod]["converged"]) < 2:
                            batch_mod_samples[mod]["converged"].append((iq, f"idx_{i}_evm_{evm:.3f}"))
                    else:
                        mod_stats[mod]["not_converged"] += 1
                        if len(batch_mod_samples[mod]["not_converged"]) < 2:
                            batch_mod_samples[mod]["not_converged"].append((iq, f"idx_{i}_evm_{evm:.3f}"))

                    total += 1

                    if total % 100 == 0 or total == selected_count:
                        not_converged = total - converged
                        print(f"\n--- Progress: {total} / {selected_count} waveforms | Converged: {converged}, Not Converged: {not_converged} ---")
                        
                        batch_dir = os.path.join(output_dir, f"batch_up_to_{total}")
                        os.makedirs(batch_dir, exist_ok=True)

                        for m, stats in sorted(mod_stats.items()):
                            print(f"  -> Modulation [{m}] => Converged: {stats['converged']} | Not Converged: {stats['not_converged']}")
                            m_samples = batch_mod_samples[m]
                            for c_idx, (iq_sample, label) in enumerate(m_samples["converged"]):
                                save_constellation_plot(iq_sample, m, f"Converged_{label}", os.path.join(batch_dir, f"converged_{c_idx+1}_{m}.png"))
                            for nc_idx, (iq_sample, label) in enumerate(m_samples["not_converged"]):
                                save_constellation_plot(iq_sample, m, f"NotConverged_{label}", os.path.join(batch_dir, f"not_converged_{nc_idx+1}_{m}.png"))

                        batch_mod_samples.clear()

            elif 'iq_data' in f and 'filename' in f:
                # Transient sliding-window format grouped by filename
                iq_data = f['iq_data']
                filenames = f['filename'][:]
                modulations = f['modulation'][:] if 'modulation' in f else None
                window_indices = f['window_index'][:] if 'window_index' in f else None

                file_to_indices = defaultdict(list)
                for idx, fname in enumerate(filenames):
                    if isinstance(fname, bytes):
                        fname = fname.decode('utf-8')
                    file_to_indices[fname].append(idx)

                unique_files = list(file_to_indices.keys())
                
                # Group files by modulation
                mod_to_files = defaultdict(list)
                for fname in unique_files:
                    first_idx = file_to_indices[fname][0]
                    if modulations is not None:
                        m = modulations[first_idx]
                        if isinstance(m, bytes):
                            m = m.decode('utf-8')
                        else:
                            m = str(m)
                    else:
                        m = parse_modulation_from_filename(fname)
                    mod_to_files[m].append(fname)
                
                selected_files = []
                for m, f_list in mod_to_files.items():
                    n_take = min(len(f_list), samples_per_mod) if samples_per_mod > 0 else len(f_list)
                    chosen = np.random.choice(f_list, n_take, replace=False)
                    selected_files.extend(chosen)
                np.random.shuffle(selected_files)

                selected_count = len(selected_files)
                batch_mod_samples = defaultdict(lambda: {"converged": [], "not_converged": []})

                for idx, fname in enumerate(selected_files):
                    indices = file_to_indices[fname]
                    if window_indices is not None:
                        indices = sorted(indices, key=lambda i: window_indices[i])
                    
                    last_indices = indices[-3:]
                    iq_chunks = []
                    for i in last_indices:
                        row = iq_data[i]
                        if row.dtype.names and 'r' in row.dtype.names and 'i' in row.dtype.names:
                            chunk = row['r'] + 1j * row['i']
                        else:
                            chunk = row[0] + 1j * row[1]
                        iq_chunks.append(chunk)
                    
                    iq_combined = np.concatenate(iq_chunks) if iq_chunks else np.array([], dtype=np.complex64)
                    if len(iq_combined) == 0:
                        continue

                    if modulations is not None:
                        mod = modulations[indices[0]]
                        if isinstance(mod, bytes):
                            mod = mod.decode('utf-8')
                        elif isinstance(mod, (np.bytes_, np.str_)):
                            mod = str(mod)
                        else:
                            mod = str(mod)
                    else:
                        mod = parse_modulation_from_filename(fname)

                    is_conv, evm = verify_convergence(iq_combined, mod, evm_threshold=evm_threshold)
                    if is_conv:
                        converged += 1
                        mod_stats[mod]["converged"] += 1
                        if len(batch_mod_samples[mod]["converged"]) < 2:
                            batch_mod_samples[mod]["converged"].append((iq_combined, f"{fname}_evm_{evm:.3f}"))
                    else:
                        mod_stats[mod]["not_converged"] += 1
                        if len(batch_mod_samples[mod]["not_converged"]) < 2:
                            batch_mod_samples[mod]["not_converged"].append((iq_combined, f"{fname}_evm_{evm:.3f}"))

                    total += 1

                    if total % 100 == 0 or total == selected_count:
                        not_converged = total - converged
                        print(f"\n--- Progress: {total} / {selected_count} recordings | Converged: {converged}, Not Converged: {not_converged} ---")
                        
                        batch_dir = os.path.join(output_dir, f"batch_up_to_{total}")
                        os.makedirs(batch_dir, exist_ok=True)

                        for m, stats in sorted(mod_stats.items()):
                            print(f"  -> Modulation [{m}] => Converged: {stats['converged']} | Not Converged: {stats['not_converged']}")
                            m_samples = batch_mod_samples[m]
                            for c_idx, (iq_sample, label) in enumerate(m_samples["converged"]):
                                save_constellation_plot(iq_sample, m, f"Converged_{label}", os.path.join(batch_dir, f"converged_{c_idx+1}_{m}.png"))
                            for nc_idx, (iq_sample, label) in enumerate(m_samples["not_converged"]):
                                save_constellation_plot(iq_sample, m, f"NotConverged_{label}", os.path.join(batch_dir, f"not_converged_{nc_idx+1}_{m}.png"))

                        batch_mod_samples.clear()
            else:
                print("Unrecognized HDF5 dataset structure.")
                return 0, 0

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    # Print final modulation breakdown
    print(f"\n=== FINAL BREAKDOWN PER MODULATION ===")
    for m, stats in sorted(mod_stats.items()):
        print(f"  -> Modulation [{m}] => Converged: {stats['converged']} | Not Converged: {stats['not_converged']}")
    print(f"=======================================")

    return total, converged


def main():
    parser = argparse.ArgumentParser(description="Check HDF5 recordings/waveforms convergence.")
    parser.add_argument("input_path", type=str, help="Path to HDF5 file")
    parser.add_argument("--evm-threshold", type=float, default=0.38, help="RMS EVM threshold")
    parser.add_argument("--samples-per-mod", type=int, default=200, help="Number of random samples to test per modulation (0 for all)")
    parser.add_argument("--output-dir", type=str, default="convergence_plots", help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    print(f"Checking HDF5 file: {args.input_path}...")
    total_recordings, total_converged = process_file(args.input_path, args.evm_threshold, samples_per_mod=args.samples_per_mod, output_dir=args.output_dir)

    total_not_converged = total_recordings - total_converged
    print(f"\nFinal Summary: {total_converged} converged, {total_not_converged} not converged out of {total_recordings} sampled items.")


if __name__ == "__main__":
    main()
