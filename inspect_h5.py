#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py

def main():
    file_path = "multi_data_channel/corrected_dataset.h5"
    print(f"Inspecting {file_path}...")
    with h5py.File(file_path, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'labels' in f:
            print("Labels shape:", f['labels'].shape)
            print("Unique labels found:", set(f['labels'][:]))
        if 'modulation_names' in f:
            print("Modulation names:", [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in f['modulation_names'][:]])
        if 'waveforms' in f:
            print("Waveforms shape:", f['waveforms'].shape)

if __name__ == '__main__':
    main()
