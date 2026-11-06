#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_transients.py

Scans transient IQ capture directories, automatically detects the signal start 
using power envelope detection (rising edge), goes back a few samples to include 
the noise floor, and generates a moving-window GIF showing the constellation lock-in.

Usage:
    python process_transients.py "C:\\Users\\Admin\\Desktop\\tests\\IQ_V5_trans"
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_iq(filename):
    iq = np.fromfile(filename, dtype=np.complex64)
    if len(iq) == 0:
        raise ValueError(f"The IQ file is empty: {filename}")
    return iq


def find_transient_start(iq, pre_margin=500, threshold_factor=15.0, smooth_window=100):
    """
    Detects the rising edge of the signal power to locate the transient start.
    Goes back `pre_margin` samples to ensure noise before the burst is captured.
    Uses a robust median-based noise floor estimator from the first 1000 samples.
    """
    power = np.abs(iq) ** 2
    # Smooth power envelope
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(power, kernel, mode='valid')
    
    # Estimate noise floor robustly using median of the first 1000 samples
    noise_sample_count = min(1000, len(smoothed) // 10)
    if noise_sample_count < 10:
        return 0
        
    noise_floor = np.median(smoothed[:noise_sample_count])
    
    # Threshold is either 15x the noise floor, or at least 0.1 absolute power
    threshold = max(noise_floor * threshold_factor, 0.1)

    # Find first index crossing the threshold
    indices = np.where(smoothed > threshold)[0]
    if len(indices) == 0:
        print("    ⚠ Warning: Transient threshold not crossed, starting at index 0.")
        return 0

    start_idx = max(0, indices[0] - pre_margin)
    return start_idx


def select_points(samples, max_points):
    if len(samples) <= max_points:
        return samples
    indices = np.linspace(0, len(samples) - 1, max_points, dtype=int)
    return samples[indices]


def make_transient_gif(iq_segment, output_file, window=512, fps=10, max_points=2000):
    """
    Creates a moving window GIF showing the constellation evolution during the transient.
    """
    N = len(iq_segment)
    if N <= window:
        starts = np.array([0])
    else:
        # Step smoothly through the segment
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
        ax.set_title(f"Transient Lock-in\nSamples: {start} → {end}")
        return scatter,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(starts),
        interval=1000 / fps,
        blit=False
    )

    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved GIF: {output_file}")


def process_file(iq_file, output_dir, duration_s=0.6):
    print(f"\nProcessing: {os.path.basename(iq_file)}")
    iq = load_iq(iq_file)

    # Determine sample rate based on modulation type in filename
    # QAM16 runs at 1,000,000 Hz, others run at 500,000 Hz
    if "QAM16" in os.path.basename(iq_file):
        samp_rate = 1000000
    else:
        samp_rate = 500000

    # Parse SPS from filename (e.g., _sps8_)
    match = re.search(r"_sps(\d+)_", os.path.basename(iq_file))
    sps = int(match.group(1)) if match else 4
    
    # Calculate window size in samples based on duration
    window_size = int(duration_s * samp_rate)
    print(f"  Parsed SPS: {sps} | Samp Rate: {samp_rate} Hz | Target Duration: {duration_s}s | Window Size: {window_size} samples")

    # Automatically find start of transient
    start_idx = find_transient_start(iq, pre_margin=500)
    print(f"  Detected transient start at sample index: {start_idx}")

    # Extract segment
    end_idx = min(start_idx + window_size, len(iq))
    iq_segment = iq[start_idx:end_idx]

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(iq_file))[0]
    output_gif = os.path.join(output_dir, f"{base_name}_transient.gif")

    # Use a moving window proportional to SPS (e.g., 128 symbols per frame)
    moving_window = min(128 * sps, len(iq_segment))
    make_transient_gif(iq_segment, output_gif, window=moving_window, fps=10)


def main():
    parser = argparse.ArgumentParser(description="Automated transient IQ extraction and GIF generation.")
    parser.add_argument("input_path", type=str, help="Path to transient IQ directory or file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for GIFs")
    parser.add_argument("--duration", type=float, default=0.6, help="Duration in seconds to extract after transient start")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Path not found: {args.input_path}")

    if os.path.isfile(args.input_path):
        iq_files = [args.input_path]
    else:
        iq_files = []
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(".bin"):
                    iq_files.append(os.path.join(root, file))

    if not iq_files:
        print(f"No .bin files found in {args.input_path}")
        return

    for iq_file in iq_files:
        if args.output_dir is None:
            parent = os.path.dirname(os.path.abspath(iq_file))
            output_dir = os.path.join(parent, "transient_gifs")
        else:
            output_dir = args.output_dir

        process_file(iq_file, output_dir, args.duration)

    print("\nAll transient files processed successfully!")


if __name__ == "__main__":
    main()
