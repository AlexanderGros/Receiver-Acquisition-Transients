# -*- coding: utf-8 -*-
# generate_h5_transient_gifs.py

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

def normalize_waveform(iq):
    """
    Normalize waveform independently to [-1, 1] preserving I/Q balance,
    matching the training pipeline normalization.
    """
    arr = np.stack([iq.real, iq.imag], axis=0)  # shape: (2, N)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized = 2 * (arr - arr_min) / (arr_max - arr_min + 1e-8) - 1
    return normalized[0] + 1j * normalized[1]

def select_points(samples, max_points):
    if len(samples) <= max_points:
        return samples
    indices = np.linspace(0, len(samples) - 1, max_points, dtype=int)
    return samples[indices]

def make_transient_gif(iq_segment, output_file, window=512, fps=10, max_points=2000, title_prefix="", normalized=False):
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

    if normalized:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    else:
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

def main():
    parser = argparse.ArgumentParser(description="Generate transient GIFs directly from an HDF5 IQ file.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input HDF5 file (e.g., trans_data_channel_3.h5)")
    parser.add_argument("--output-dir", type=str, default="gifs_output", help="Directory to save the generated GIFs")
    parser.add_argument("--num-waveforms", type=int, default=5, help="Number of waveforms to process (-1 for all)")
    parser.add_argument("--normalize", action="store_true", help="Apply per-waveform [-1, 1] normalization before plotting")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input_file, 'r') as f:
        waveforms_key = 'iq_data' if 'iq_data' in f else next((k for k in ['waveforms', 'data', 'X', 'iq', 'signals'] if k in f), None)
        if waveforms_key is None:
            raise ValueError(f"Could not find waveform dataset in {args.input_file}")

        waveforms_ds = f[waveforms_key]
        total_waveforms = waveforms_ds.shape[0]

        filenames = f['filename'][:] if 'filename' in f else None
        modulations = f['modulation'][:] if 'modulation' in f else None
        sps_arr = f['sps'][:] if 'sps' in f else None

        if args.num_waveforms == -1 or args.num_waveforms >= total_waveforms:
            indices = range(total_waveforms)
        else:
            indices = np.random.choice(total_waveforms, size=args.num_waveforms, replace=False)

        print(f"Processing {len(indices)} waveform(s) from {args.input_file} (Normalize={args.normalize})...")

        for idx in indices:
            wf = waveforms_ds[idx]

            if wf.dtype.names and 'r' in wf.dtype.names and 'i' in wf.dtype.names:
                iq = wf['r'] + 1j * wf['i']
            elif wf.ndim == 2 and wf.shape[0] == 2:
                iq = wf[0] + 1j * wf[1]
            else:
                iq = wf
            iq = iq.astype(np.complex64)

            if args.normalize:
                iq = normalize_waveform(iq)

            file_name = filenames[idx].decode('utf-8') if filenames is not None else f"waveform_{idx}"
            mod_val = modulations[idx].decode('utf-8') if modulations is not None else "UNKNOWN"
            sps = int(sps_arr[idx]) if sps_arr is not None else 4

            safe_name = os.path.splitext(os.path.basename(file_name))[0]
            suffix = "_normalized" if args.normalize else ""
            output_file = os.path.join(args.output_dir, f"{safe_name}_{mod_val}{suffix}_transient.gif")

            samp_rate = 1000000 if "QAM16" in mod_val.upper() else 500000
            start_idx = find_transient_start(iq, pre_margin=500)
            window_size = int(0.6 * samp_rate)
            iq_segment = iq[start_idx:min(start_idx + window_size, len(iq))]

            moving_window = min(128 * sps, len(iq_segment))
            print(f"  -> Generating GIF for waveform {idx} ({mod_val}): {output_file}")
            make_transient_gif(iq_segment, output_file, window=moving_window, fps=10, title_prefix=f"{mod_val} (SPS={sps})", normalized=args.normalize)

    print("\nAll GIFs generated successfully!")

if __name__ == "__main__":
    main()
