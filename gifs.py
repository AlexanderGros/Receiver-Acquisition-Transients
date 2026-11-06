#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_iq.py

Offline analysis of a GNU Radio complex IQ binary file.

Input:
    GNU Radio file_sink containing complex64 IQ samples, or a directory
    containing subfolders with IQ files.

Outputs:
    1. Three PNG constellation plots:
       - beginning
       - middle
       - end

    2. GIF animation showing the evolution of the constellation
       through the capture.

Usage example:

    python gifs.py "C:\\Users\\Admin\\Desktop\\tests\\IQ"

Optional arguments:

    --window 5000
        Number of IQ samples displayed in each constellation frame.

    --frames 30
        Number of frames in the GIF.

    --output-dir results
        Directory where plots/GIFs are saved.

    --fps 5
        Animation speed in frames per second.

    --max-points 5000
        Maximum number of points displayed in a constellation.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ══════════════════════════════════════════════════════════════════════════════
# LOAD IQ FILE
# ══════════════════════════════════════════════════════════════════════════════

def load_iq(filename):
    """
    Load GNU Radio complex float32 IQ file.

    GNU Radio:
        gr.sizeof_gr_complex = 8 bytes
        complex64 = float32 I + float32 Q
    """

    print(f"\nLoading IQ file:")
    print(f"  {filename}")

    iq = np.fromfile(filename, dtype=np.complex64)

    print(f"  Samples: {len(iq):,}")
    print(f"  File size: {os.path.getsize(filename):,} bytes")

    if len(iq) == 0:
        raise ValueError("The IQ file contains no samples.")

    return iq


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize(iq):
    """
    Normalize constellation amplitude.

    This is only for visualization.
    It does NOT modify the original IQ data.
    """

    amplitude = np.sqrt(np.mean(np.abs(iq) ** 2))

    if amplitude == 0:
        return iq

    return iq / amplitude


# ══════════════════════════════════════════════════════════════════════════════
# SELECT POINTS FOR DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def select_points(samples, max_points):
    """
    Reduce the number of displayed points if necessary.

    The actual IQ data is never modified.
    """

    if len(samples) <= max_points:
        return samples

    indices = np.linspace(
        0,
        len(samples) - 1,
        max_points,
        dtype=int
    )

    return samples[indices]


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CONSTELLATION PLOT
# ══════════════════════════════════════════════════════════════════════════════

def save_constellation(samples, title, filename, max_points):

    samples = select_points(samples, max_points)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        samples.real,
        samples.imag,
        s=3,
        alpha=0.5
    )

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)

    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(title)

    plt.axis("equal")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# THREE STATIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_three_plots(iq, output_dir, window, max_points):

    print("\nCreating three constellation plots...")

    N = len(iq)

    # Beginning
    begin_start = 0

    # Middle
    middle_start = max(0, N // 2 - window // 2)

    # End
    end_start = max(0, N - window)

    positions = [
        ("beginning", begin_start),
        ("middle", middle_start),
        ("end", end_start),
    ]

    for name, start in positions:

        samples = iq[start:start + window]

        title = (
            f"{name.capitalize()} "
            f"(samples {start:,} → {start + len(samples):,})"
        )

        filename = os.path.join(
            output_dir,
            f"constellation_{name}.png"
        )

        save_constellation(
            samples,
            title,
            filename,
            max_points
        )


# ══════════════════════════════════════════════════════════════════════════════
# GIF — PROGRESSIVE CONSTELLATION
# ══════════════════════════════════════════════════════════════════════════════

def make_progressive_gif(
    iq,
    output_file,
    window,
    frames,
    fps,
    max_points
):
    """
    Each frame contains the IQ samples from the beginning
    of the capture up to the current point.

    This shows the progressive evolution of the constellation.
    """

    print("\nCreating progressive GIF...")

    N = len(iq)

    if N < window:
        window = N

    # Positions through the file
    positions = np.linspace(
        window,
        N,
        frames,
        dtype=int
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)

    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)

    # Fixed limits based on the entire signal
    max_amp = np.max(np.abs(iq))

    if max_amp == 0:
        max_amp = 1

    limit = max_amp * 1.1

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    scatter = ax.scatter([], [], s=3, alpha=0.5)

    def update(frame):

        end = positions[frame]

        samples = iq[:end]

        samples = select_points(samples, max_points)

        scatter.set_offsets(
            np.column_stack(
                (samples.real, samples.imag)
            )
        )

        ax.set_title(
            f"Progressive constellation\n"
            f"{end:,} / {N:,} samples"
        )

        return scatter,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=False
    )

    animation.save(
        output_file,
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print(f"  Saved: {output_file}")


# ══════════════════════════════════════════════════════════════════════════════
# GIF — MOVING WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def make_moving_window_gif(
    iq,
    output_file,
    window,
    frames,
    fps,
    max_points
):
    """
    Each frame shows only a moving window of IQ samples.

    This is particularly useful for observing receiver convergence,
    because old samples do not remain visible.
    """

    print("\nCreating moving-window GIF...")

    N = len(iq)

    if N <= window:
        starts = np.array([0])
    else:
        starts = np.linspace(
            0,
            N - window,
            frames,
            dtype=int
        )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.grid(True)

    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)

    # Fixed constellation limits
    max_amp = np.max(np.abs(iq))

    if max_amp == 0:
        max_amp = 1

    limit = max_amp * 1.1

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    scatter = ax.scatter([], [], s=3, alpha=0.5)

    def update(frame):

        start = starts[frame]
        end = min(start + window, N)

        samples = iq[start:end]

        samples = select_points(samples, max_points)

        scatter.set_offsets(
            np.column_stack(
                (samples.real, samples.imag)
            )
        )

        ax.set_title(
            f"Moving window\n"
            f"samples {start:,} → {end:,}"
        )

        return scatter,

    animation = FuncAnimation(
        fig,
        update,
        frames=len(starts),
        interval=1000 / fps,
        blit=False
    )

    animation.save(
        output_file,
        writer=PillowWriter(fps=fps)
    )

    plt.close(fig)

    print(f"  Saved: {output_file}")


def process_single_file(iq_file, output_dir, window, frames, fps, max_points):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "═" * 70)
    print(f"OFFLINE IQ ANALYSIS: {os.path.basename(iq_file)}")
    print("═" * 70)

    print(f"Input : {iq_file}")
    print(f"Output: {output_dir}")

    iq = load_iq(iq_file)

    make_three_plots(
        iq,
        output_dir,
        window,
        max_points
    )

    progressive_file = os.path.join(
        output_dir,
        "constellation_progressive.gif"
    )

    make_progressive_gif(
        iq,
        progressive_file,
        window,
        frames,
        fps,
        max_points
    )

    moving_file = os.path.join(
        output_dir,
        "constellation_moving_window.gif"
    )

    make_moving_window_gif(
        iq,
        moving_file,
        window,
        frames,
        fps,
        max_points
    )

    print("\n" + "═" * 70)
    print("DONE")
    print("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    parser = argparse.ArgumentParser(
        description="Offline visualization of GNU Radio IQ binary files or directories."
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to GNU Radio complex64 IQ binary file or directory containing subfolders with IQ files"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5000,
        help="Number of IQ samples per constellation window"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames in each GIF"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5,
        help="GIF playback speed"
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum number of points displayed per frame"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (or base output directory when processing a folder)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
            f"Input path not found:\n{args.input_path}"
        )

    if os.path.isfile(args.input_path):
        iq_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        modulation_dict = {}
        for root, dirs, files in os.walk(args.input_path):
            bin_files = [os.path.join(root, f) for f in files if f.endswith(".bin")]
            if bin_files:
                rel = os.path.relpath(root, args.input_path)
                mod = rel.split(os.sep)[0] if rel != '.' else 'default'
                if mod not in modulation_dict:
                    modulation_dict[mod] = []
                modulation_dict[mod].extend(bin_files)
        
        iq_files = []
        for mod, files in modulation_dict.items():
            files.sort()
            iq_files.extend(files[:5])

        if not iq_files:
            print(f"No .bin files found in subfolders of: {args.input_path}")
            return
    else:
        raise ValueError(f"Invalid input path: {args.input_path}")

    for iq_file in iq_files:
        if args.output_dir is None:
            base = os.path.splitext(
                os.path.basename(iq_file)
            )[0]
            parent = os.path.dirname(
                os.path.abspath(iq_file)
            )
            output_dir = os.path.join(
                parent,
                base + "_analysis"
            )
        else:
            if os.path.isdir(args.input_path):
                rel_path = os.path.relpath(os.path.dirname(iq_file), args.input_path)
                base = os.path.splitext(os.path.basename(iq_file))[0]
                output_dir = os.path.join(args.output_dir, rel_path, base + "_analysis")
            else:
                output_dir = args.output_dir

        process_single_file(
            iq_file,
            output_dir,
            args.window,
            args.frames,
            args.fps,
            args.max_points
        )


if __name__ == "__main__":
    main()
