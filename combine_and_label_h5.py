#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_and_label_h5.py

Scans a directory for individual .h5 waveform files, parses their modulation type 
and whether they are raw or corrected based on their filenames, and combines them 
into two master H5 datasets: `raw_dataset.h5` and `corrected_dataset.h5`.

Usage:
    python combine_and_label_h5.py "C:\\Users\\Admin\\Desktop\\tests\\IQ_V5_multi"
"""

import os
import re
import argparse
import h5py
import numpy as np

# Define standard modulation mapping to integer labels
MODULATION_MAP = {
    "BPSK": 0,
    "QPSK": 1,
    "8PSK": 2,
    "QAM16": 3
}

def parse_metadata(filename):
    basename = os.path.basename(filename)
    
    # Identify modulation
    modulation = "UNKNOWN"
    for mod in MODULATION_MAP.keys():
        if basename.startswith(mod):
            modulation = mod
            break
            
    # Check if corrected
    is_corrected = "corrected" in basename.lower()
    
    return modulation, is_corrected

def main():
    parser = argparse.ArgumentParser(description="Combine individual H5 files into raw and corrected master datasets.")
    parser.add_argument("input_dir", type=str, help="Path to directory containing individual .h5 files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save master H5 files (defaults to input_dir)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    output_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    h5_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".h5") and not file.endswith("_dataset.h5"):
                h5_files.append(os.path.join(root, file))

    if not h5_files:
        print(f"No individual .h5 files found in {args.input_dir}")
        return

    print(f"Found {len(h5_files)} .h5 files. Categorizing and combining...\n")

    raw_waveforms = []
    raw_labels = []
    raw_mod_names = []

    corrected_waveforms = []
    corrected_labels = []
    corrected_mod_names = []

    for h5_path in h5_files:
        modulation, is_corrected = parse_metadata(h5_path)
        
        if modulation not in MODULATION_MAP:
            print(f"  ⚠ Skipping {os.path.basename(h5_path)}: Unknown modulation.")
            continue

        label = MODULATION_MAP[modulation]

        try:
            with h5py.File(h5_path, 'r') as f:
                if 'waveforms' not in f:
                    print(f"  ⚠ Skipping {os.path.basename(h5_path)}: 'waveforms' dataset missing.")
                    continue
                data = f['waveforms'][:] # Shape expected: (num_waveforms, 2, waveform_length)
                
            if is_corrected:
                corrected_waveforms.append(data)
                corrected_labels.extend([label] * data.shape[0])
                corrected_mod_names.extend([modulation] * data.shape[0])
            else:
                raw_waveforms.append(data)
                raw_labels.extend([label] * data.shape[0])
                raw_mod_names.extend([modulation] * data.shape[0])
                
            print(f"  ✓ Processed: {os.path.basename(h5_path)} ({'Corrected' if is_corrected else 'Raw'}, {modulation})")

        except Exception as e:
            print(f"  ✗ Error reading {h5_path}: {e}")

    dt_str = h5py.string_dtype(encoding='utf-8')

    # Save Raw Master Dataset
    if raw_waveforms:
        raw_x = np.concatenate(raw_waveforms, axis=0)
        raw_y = np.array(raw_labels, dtype=np.int32)
        raw_names = np.array(raw_mod_names, dtype=dt_str)
        
        raw_output_path = os.path.join(output_dir, "raw_dataset.h5")
        with h5py.File(raw_output_path, 'w') as f:
            f.create_dataset('waveforms', data=raw_x, compression="gzip")
            f.create_dataset('labels', data=raw_y)
            f.create_dataset('modulation_names', data=raw_names)
            
        print(f"\nSaved Raw Dataset -> {raw_output_path}")
        print(f"  Shape: {raw_x.shape}, Labels shape: {raw_y.shape}")
    else:
        print("\nNo raw waveform files were successfully processed.")

    # Save Corrected Master Dataset
    if corrected_waveforms:
        corr_x = np.concatenate(corrected_waveforms, axis=0)
        corr_y = np.array(corrected_labels, dtype=np.int32)
        corr_names = np.array(corrected_mod_names, dtype=dt_str)
        
        corr_output_path = os.path.join(output_dir, "corrected_dataset.h5")
        with h5py.File(corr_output_path, 'w') as f:
            f.create_dataset('waveforms', data=corr_x, compression="gzip")
            f.create_dataset('labels', data=corr_y)
            f.create_dataset('modulation_names', data=corr_names)
            
        print(f"\nSaved Corrected Dataset -> {corr_output_path}")
        print(f"  Shape: {corr_x.shape}, Labels shape: {corr_y.shape}")
    else:
        print("\nNo corrected waveform files were successfully processed.")

    print("\nMaster H5 dataset generation completed!")

if __name__ == '__main__':
    main()
