#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONVERT_BIN_TO_H5.py — Script séparé de conversion des fichiers .bin en .h5

Parcourt le répertoire de base des captures IQ, trouve tous les fichiers .bin,
et les convertit en fichiers .h5 avec la forme (num_waveforms, 2, waveform_length).
"""

import os
import argparse
import numpy as np
import h5py

def convert_bin_to_h5(bin_path, h5_path, num_waveforms, waveform_length):
    if not os.path.exists(bin_path):
        print(f"    ✗ Fichier binaire introuvable : {bin_path}")
        return False
    
    file_size = os.path.getsize(bin_path)
    total_samples = file_size // 8
    required_samples = num_waveforms * waveform_length

    if total_samples < required_samples:
        print(f"    ⚠ Pas assez d'échantillons ({total_samples} < {required_samples}) dans {os.path.basename(bin_path)}.")
        usable_waveforms = total_samples // waveform_length
        if usable_waveforms <= 0:
            print("    ✗ Trop peu de données pour former au moins une waveform.")
            return False
        samples_to_read = usable_waveforms * waveform_length
    else:
        usable_waveforms = num_waveforms
        samples_to_read = required_samples

    # Lecture des données binaires en complex64
    data = np.fromfile(bin_path, dtype=np.complex64, count=samples_to_read)
    
    # Reshape en (usable_waveforms, waveform_length)
    data = data.reshape((usable_waveforms, waveform_length))
    
    # Création du tableau de forme (usable_waveforms, 2, waveform_length)
    # Canal 0 : Real (I), Canal 1 : Imag (Q)
    h5_data = np.zeros((usable_waveforms, 2, waveform_length), dtype=np.float32)
    h5_data[:, 0, :] = data.real
    h5_data[:, 1, :] = data.imag

    # Écriture dans le fichier HDF5
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('waveforms', data=h5_data, compression="gzip")

    print(f"    ✓ Converti : {os.path.basename(bin_path)} → {os.path.basename(h5_path)} (shape: {h5_data.shape})")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert .bin IQ files to .h5")
    parser.add_argument("--iq-base-dir", type=str, default="/mnt/cephfs/proj/umons/amcssa/testbench_analysis/multi_data_channel",
                        help="Base directory containing .bin files")
    parser.add_argument("--num-waveforms", type=int, default=2000,
                        help="Number of waveforms per file")
    parser.add_argument("--waveform-length", type=int, default=65536,
                        help="Length of each waveform")
    
    args = parser.parse_args()

    iq_base_dir = args.iq_base_dir
    num_waveforms = args.num_waveforms
    waveform_length = args.waveform_length

    print(f"Recherche des fichiers .bin dans : {iq_base_dir}\n")
    if not os.path.exists(iq_base_dir):
        print(f"Erreur : Le dossier {iq_base_dir} n'existe pas.")
        return

    converted_count = 0
    total_files = 0

    for root, dirs, files in os.walk(iq_base_dir):
        for file in files:
            if file.endswith(".bin"):
                total_files += 1
                bin_path = os.path.join(root, file)
                h5_path = os.path.splitext(bin_path)[0] + ".h5"
                
                print(f"Traitement de : {file}")
                if convert_bin_to_h5(bin_path, h5_path, num_waveforms, waveform_length):
                    converted_count += 1

    print(f"\nConversion terminée : {converted_count}/{total_files} fichiers convertis avec succès en H5.")

if __name__ == '__main__':
    main()
