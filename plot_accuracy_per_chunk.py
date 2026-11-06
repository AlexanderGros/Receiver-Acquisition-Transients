# -*- coding: utf-8 -*-
# plot_accuracy_per_chunk.py

import os
import re
import glob
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_iq(iq_path, file_name=None):
    if iq_path.endswith('.h5'):
        with h5py.File(iq_path, 'r') as f:
            waveforms_key = 'iq_data' if 'iq_data' in f else next((k for k in ['waveforms', 'data', 'X', 'iq', 'signals'] if k in f), None)
            if waveforms_key is None:
                keys = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
                if not keys:
                    raise ValueError(f"No datasets found in HDF5 file: {iq_path}.")
                waveforms_key = keys[0]

            waveforms = f[waveforms_key]
            idx = 0
            if file_name is not None and 'filename' in f:
                filenames = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in f['filename'][:]]
                if file_name in filenames:
                    idx = filenames.index(file_name)
                elif file_name.isdigit():
                    idx = int(file_name)
            
            wf = waveforms[idx]
            
            # Handle compound dtype {'r', 'i'} present in iq_data
            if wf.dtype.names and 'r' in wf.dtype.names and 'i' in wf.dtype.names:
                iq = wf['r'] + 1j * wf['i']
            elif wf.ndim == 2 and wf.shape[0] == 2:
                iq = wf[0] + 1j * wf[1]
            else:
                iq = wf
            return iq.astype(np.complex64)
    else:
        iq = np.fromfile(iq_path, dtype=np.complex64)
        if len(iq) == 0:
            raise ValueError(f"The IQ file is empty: {iq_path}")
        return iq

def find_transient_start(iq, pre_margin=500, threshold_factor=15.0, smooth_window=100):
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

def select_points(samples, max_points):
    if len(samples) <= max_points:
        return samples
    indices = np.linspace(0, len(samples) - 1, max_points, dtype=int)
    return samples[indices]

def make_transient_gif(iq_segment, output_file, window=512, fps=10, max_points=2000, title_prefix=""):
    N = len(iq_segment)
    if N <= window:
        starts = np.array([0])
    else:
        frames = 40
        starts = np.linspace(0, N - window, frames, dtype=int)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)
    ax.axhline(0, linewidth=0.5, color='gray')
    ax.axvline(0, linewidth=0.5, color='gray')

    max_amp = np.max(np.abs(iq_segment))
    if max_amp == 0:
        max_amp = 1
    limit = max_amp * 1.1

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    scatter = ax.scatter([], [], s=4, alpha=0.6)

    def update(frame):
        start = starts[frame]
        end = min(start + window, N)
        samples = iq_segment[start:end]
        samples = select_points(samples, max_points)

        scatter.set_offsets(np.column_stack((samples.real, samples.imag)))
        ax.set_title(f"{title_prefix}\nSamples: {start} → {end}")
        return scatter,

    animation = FuncAnimation(fig, update, frames=len(starts), interval=1000 / fps, blit=False)
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)

def sample_waveforms(df, num_samples, seed=42):
    selected = []
    np.random.seed(seed)
    if 'filename' not in df.columns:
        return []
    
    if 'true_modulation' in df.columns:
        modulations = df['true_modulation'].unique()
        for mod in modulations:
            df_mod = df[df['true_modulation'] == mod]
            mod_files = df_mod['filename'].unique()
            if num_samples != -1 and len(mod_files) > num_samples:
                sel = np.random.choice(mod_files, size=num_samples, replace=False)
            else:
                sel = mod_files
            selected.extend(sel)
    else:
        unique_files = df['filename'].unique()
        if num_samples != -1 and len(unique_files) > num_samples:
            selected = np.random.choice(unique_files, size=num_samples, replace=False).tolist()
        else:
            selected = unique_files.tolist() if hasattr(unique_files, 'tolist') else list(unique_files)
    return selected

def main():
    parser = argparse.ArgumentParser(description="Plot classification accuracy vs. chunk and generate transient GIFs.")
    parser.add_argument("--input-dir", type=str, default="output_3", help="Directory containing the result CSV files")
    parser.add_argument("--iq-path", type=str, default=None, help="Path to raw .bin directory or specific .h5 IQ file")
    parser.add_argument("--results-pattern", type=str, default="results_*.csv", help="Glob pattern to find result CSV files")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save the generated plots and GIFs")
    parser.add_argument("--num-avg-waveforms", type=int, default=10, help="Number of waveforms per modulation to aggregate for average plots (-1 for all)")
    parser.add_argument("--num-indiv-waveforms", type=int, default=3, help="Number of waveforms to plot individually and generate GIFs (-1 for all)")
    args = parser.parse_args()

    search_path = os.path.join(args.input_dir, args.results_pattern)
    csv_files = glob.glob(search_path)
    if not csv_files:
        raise FileNotFoundError(f"No result CSV files found matching pattern: {search_path}.")

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in csv_files:
        print(f"\nProcessing results from {csv_path}...")
        df = pd.read_csv(csv_path)

        base_name = os.path.basename(csv_path)
        parts = base_name.replace("results_", "").replace(".csv", "").rsplit("_", 1)
        model_name = parts[0] if len(parts) > 0 else "unknown_model"
        weights_type = parts[1] if len(parts) > 1 else "unknown_weights"

        if 'filename' not in df.columns:
            print("Warning: 'filename' column not found. Skipping individual plots/GIFs.")
            continue

        avg_files = sample_waveforms(df, args.num_avg_waveforms, seed=42)
        df_avg = df[df['filename'].isin(avg_files)] if avg_files else df
        avg_suffix = f"per_mod_{args.num_avg_waveforms}" if args.num_avg_waveforms != -1 else "all"
        title_avg_suffix = f"({args.num_avg_waveforms} Waveforms/Mod)" if args.num_avg_waveforms != -1 else "(All Waveforms)"

        indiv_files = sample_waveforms(df, args.num_indiv_waveforms, seed=42)
        df_indiv = df[df['filename'].isin(indiv_files)] if indiv_files else df

        # 1. Combined averaged plot
        accuracy_per_window = df_avg.groupby('window_index')['correct'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_per_window['window_index'], accuracy_per_window['correct'], marker='o', linestyle='-', color='tab:blue')
        plt.title(f"[{model_name.upper()} - {weights_type}] Accuracy vs. Transient Chunk (All Mods) {title_avg_suffix}")
        plt.xlabel("Chunk / Window Index")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, f"{model_name}_{weights_type}_acc_vs_chunks_ALL_mods_{avg_suffix}.png"))
        plt.close()

        # 2. Per-modulation averaged plots
        if 'true_modulation' in df_avg.columns:
            for mod in df_avg['true_modulation'].unique():
                df_mod = df_avg[df_avg['true_modulation'] == mod]
                if len(df_mod) == 0: continue
                acc_mod = df_mod.groupby('window_index')['correct'].mean().reset_index()

                plt.figure(figsize=(10, 5))
                plt.plot(acc_mod['window_index'], acc_mod['correct'], marker='s', linestyle='-', color='tab:orange')
                plt.title(f"[{model_name.upper()} - {weights_type}] Accuracy vs. Transient Chunk ({mod}) {title_avg_suffix}")
                plt.xlabel("Chunk / Window Index")
                plt.ylabel("Accuracy")
                plt.ylim(0, 1.05)
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, f"{model_name}_{weights_type}_acc_vs_chunks_{mod}_{avg_suffix}.png"))
                plt.close()

        # 3. Individual waveform plots & Transient GIFs
        if len(indiv_files) > 0:
            print(f"  -> Generating individual waveform plots and GIFs for {len(indiv_files)} file(s)...")
            for file_name in indiv_files:
                df_file = df_indiv[df_indiv['filename'] == file_name]
                if len(df_file) == 0: continue

                acc_file = df_file.groupby('window_index')['correct'].mean().reset_index()
                mod_val = df_file['true_modulation'].iloc[0] if 'true_modulation' in df_file.columns else 'UNKNOWN'
                safe_file_name = os.path.splitext(file_name)[0]

                # Plot accuracy
                plt.figure(figsize=(10, 5))
                plt.plot(acc_file['window_index'], acc_file['correct'], marker='^', linestyle='-', color='tab:green')
                plt.title(f"[{model_name.upper()} - {weights_type}] Individual Waveform Accuracy ({mod_val}) - {safe_file_name}")
                plt.xlabel("Chunk / Window Index")
                plt.ylabel("Accuracy (Correct / Incorrect)")
                plt.ylim(-0.05, 1.05)
                plt.grid(True)
                plt.savefig(os.path.join(args.output_dir, f"{model_name}_{weights_type}_indiv_{safe_file_name}_acc_vs_chunks.png"))
                plt.close()

                # Generate Transient GIF if --iq-path is provided
                if args.iq_path:
                    try:
                        if args.iq_path.endswith('.h5'):
                            iq = load_iq(args.iq_path, file_name=file_name)
                        else:
                            iq_path = os.path.join(args.iq_path, file_name)
                            if not os.path.exists(iq_path):
                                continue
                            iq = load_iq(iq_path)

                        samp_rate = 1000000 if "QAM16" in file_name else 500000
                        match = re.search(r"_sps(\d+)_", file_name)
                        sps = int(match.group(1)) if match else 4
                        
                        start_idx = find_transient_start(iq, pre_margin=500)
                        window_size = int(0.6 * samp_rate)
                        iq_segment = iq[start_idx:min(start_idx + window_size, len(iq))]
                        
                        moving_window = min(128 * sps, len(iq_segment))
                        gif_path = os.path.join(args.output_dir, f"{safe_file_name}_transient.gif")
                        make_transient_gif(iq_segment, gif_path, window=moving_window, fps=10, title_prefix=f"Transient Lock-in ({mod_val})")
                        print(f"     -> Saved transient GIF: {gif_path}")
                    except Exception as e:
                        print(f"     ⚠ Warning: Could not generate GIF for {file_name}: {e}")

    print("\nAll plots and GIFs generated successfully!")

if __name__ == "__main__":
    main()
