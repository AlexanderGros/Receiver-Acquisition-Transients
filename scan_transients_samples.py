#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_transients_samples.py

Scans a directory of transient IQ capture files (.bin), parses their modulation type,
calculates their sample counts (ignoring files with 0 samples), and reports the 
minimum number of samples per modulation as well as overall.

Usage:
    python scan_transients_samples.py "C:\\Users\\Admin\\Desktop\\tests\\IQ_V5_trans"
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict


def parse_modulation(filename):
    basename = os.path.basename(filename)
    mod_match = re.match(r"^([^_]+)", basename)
    return mod_match.group(1) if mod_match else "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Scan transient IQ files and find minimum sample counts per modulation.")
    parser.add_argument("input_dir", type=str, help="Path to root transient IQ directory")
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

    print(f"Scanning {len(iq_files)} .bin files...\n")

    mod_stats = defaultdict(list)
    zero_sample_files = 0
    total_valid_files = 0

    for iq_file in iq_files:
        try:
            # Complex64 = 8 bytes per sample
            file_size = os.path.getsize(iq_file)
            num_samples = file_size // 8

            if num_samples == 0:
                zero_sample_files += 1
                continue

            modulation = parse_modulation(iq_file)
            mod_stats[modulation].append((num_samples, iq_file))
            total_valid_files += 1

        except Exception as e:
            print(f"Error reading {iq_file}: {e}")

    print("=" * 60)
    print(f"SCAN RESULTS (Total valid files: {total_valid_files}, Ignored zero-sample files: {zero_sample_files})")
    print("=" * 60)

    global_min = float('inf')

    for mod, entries in sorted(mod_stats.items()):
        samples_list = [entry[0] for entry in entries]
        min_samp = min(samples_list)
        max_samp = max(samples_list)
        avg_samp = sum(samples_list) / len(samples_list)

        if min_samp < global_min:
            global_min = min_samp

        print(f"Modulation: {mod}")
        print(f"  - File count    : {len(entries)}")
        print(f"  - Min samples   : {min_samp:,}")
        print(f"  - Max samples   : {max_samp:,}")
        print(f"  - Mean samples  : {int(avg_samp):,}")
        print("-" * 60)

    if global_min != float('inf'):
        print(f"\nGlobal Minimum Samples across all modulations: {global_min:,}")
    else:
        print("\nNo valid samples found across any files.")


if __name__ == "__main__":
    main()
