# -*- coding: utf-8 -*-
# plot_threshold_windows.py

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot real and imaginary parts of concatenated transient windows from HDF5 dataset per modulation and run around the threshold.")
    parser.add_argument("--dataset", type=str, default="trans_data_channel_2.h5", help="Path to transient HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default="threshold_plotter", help="Output directory for plots")
    parser.add_argument("--samples-per-mod", type=int, default=10, help="Number of different runs/files to plot per modulation")
    parser.add_argument("--full-waveform", action="store_true", help="Plot the entire waveform instead of filtering around the threshold windows")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found at {args.dataset}")

    with h5py.File(args.dataset, 'r') as f:
        iq_dataset = f['iq_data']
        filenames = f['filename'][:]
        window_indices = f['window_index'][:]
        modulations = f['modulation'][:]
        run_indices = f['run_index'][:] if 'run_index' in f else np.zeros(len(filenames), dtype=np.int32)

        total_entries = len(filenames)
        if total_entries == 0:
            print("Dataset is empty.")
            return

        # Decode strings
        decoded_mods = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in modulations]
        decoded_files = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn) for fn in filenames]

        unique_mods = sorted(list(set(decoded_mods)))

        plot_counter = 0
        for mod in unique_mods:
            print(f"Processing modulation: {mod}")
            
            # Find indices for this modulation
            mod_mask = np.array([m == mod for m in decoded_mods])
            mod_indices = np.where(mod_mask)[0]

            if len(mod_indices) == 0:
                continue

            # Find unique files/runs for this modulation
            mod_files = [decoded_files[i] for i in mod_indices]
            mod_runs = [run_indices[i] for i in mod_indices]
            
            # Pair file with run index to get distinct runs
            file_run_pairs = list(set(zip(mod_files, mod_runs)))
            # Limit to samples-per-mod
            selected_pairs = file_run_pairs[:args.samples_per_mod]

            for target_file, target_run in selected_pairs:
                # Get indices for this specific file/run
                file_mask = np.array([(decoded_files[i] == target_file and run_indices[i] == target_run) for i in mod_indices])
                indices = mod_indices[file_mask]

                if len(indices) == 0:
                    continue

                # Sort indices by window_index to maintain correct temporal order
                file_window_indices = window_indices[indices]
                sorted_order = np.argsort(file_window_indices)
                indices = indices[sorted_order]
                file_window_indices = file_window_indices[sorted_order]

                target_transient_w_idx = 10

                if not args.full_waveform:
                    # Filter to take 2 windows before and 4 windows after the threshold start (window_index == 10)
                    min_w = target_transient_w_idx - 2
                    max_w = target_transient_w_idx + 4
                    
                    window_mask = (file_window_indices >= min_w) & (file_window_indices <= max_w)
                    indices = indices[window_mask]
                    file_window_indices = file_window_indices[window_mask]

                    if len(indices) == 0:
                        continue

                # Extract windows for this file
                file_iq_windows = iq_dataset[indices]  # shape: (num_windows, 4096)

                # Concatenate windows to view them continuously
                segment = file_iq_windows.flatten()
                real_part = segment.real
                imag_part = segment.imag

                plt.figure(figsize=(14, 5))
                plt.plot(real_part, label='Real', alpha=0.7, color='blue')
                plt.plot(imag_part, label='Imaginary', alpha=0.7, color='orange')

                # Draw vertical lines at window boundaries (every 4096 samples)
                window_size = 4096
                for w in range(1, len(indices)):
                    x_pos = w * window_size
                    actual_w_idx = file_window_indices[w]
                    line_color = 'red' if actual_w_idx == target_transient_w_idx else 'gray'
                    line_style = '--' if actual_w_idx == target_transient_w_idx else ':'
                    plt.axvline(x=x_pos, color=line_color, linestyle=line_style, alpha=0.8)

                if args.full_waveform:
                    plt.title(f"Modulation: {mod} | Run: {target_run} | File: {target_file} | Full Waveform (Windows: {len(indices)})")
                else:
                    plt.title(f"Modulation: {mod} | Run: {target_run} | File: {target_file} | Windows: {min_w} to {max_w}")
                
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.tight_layout()

                out_filename = f"plot_{mod}_run{target_run}_{plot_counter}.png"
                out_path = os.path.join(args.output_dir, out_filename)
                plt.savefig(out_path)
                plt.close()
                print(f"Saved plot to {out_path}")

                # Plot constellation diagram for the last up to 10,000 samples of the segment
                constellation_samples = segment[-10000:]
                plt.figure(figsize=(6, 6))
                plt.scatter(constellation_samples.real, constellation_samples.imag, s=1, alpha=0.5, color='purple')
                plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
                plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
                plt.title(f"Constellation (Last Samples): {mod} | Run: {target_run}")
                plt.xlabel("In-Phase (I)")
                plt.ylabel("Quadrature (Q)")
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()

                const_filename = f"constellation_{mod}_run{target_run}_{plot_counter}.png"
                const_path = os.path.join(args.output_dir, const_filename)
                plt.savefig(const_path)
                plt.close()
                print(f"Saved constellation plot to {const_path}")

                plot_counter += 1

    print("Plotting completed successfully!")

if __name__ == "__main__":
    main()
