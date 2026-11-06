#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_h5_constellations.py — Script pour lire les fichiers .h5 et tracer les constellations IQ.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

IQ_BASE_DIR = r"C:\Users\Admin\Desktop\tests\IQ_V5_multi"

def main():
    if not os.path.exists(IQ_BASE_DIR):
        print(f"Erreur : Le dossier {IQ_BASE_DIR} n'existe pas.")
        return

    h5_files = []
    for root, dirs, files in os.walk(IQ_BASE_DIR):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    if not h5_files:
        print("Aucun fichier .h5 trouvé dans le répertoire.")
        return

    print(f"Trouvé {len(h5_files)} fichiers H5. Affichage des constellations...")

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'waveforms' not in f:
                    print(f"  ⚠ Dataset 'waveforms' absent dans {os.path.basename(h5_path)}")
                    continue
                h5_data = f['waveforms'][:] # Shape attendue : (num_waveforms, 2, waveform_length)

            # Reconstruction du signal complexe : I + j*Q
            # Canal 0 : Real (I), Canal 1 : Imag (Q)
            iq_complex = h5_data[:, 0, :] + 1j * h5_data[:, 1, :]
            iq_flat = iq_complex.flatten()

            # Tracé de la constellation
            plt.figure(figsize=(6, 6))
            plt.scatter(iq_flat.real, iq_flat.imag, s=1, alpha=0.5, color='blue')
            plt.title(f"Constellation: {os.path.basename(h5_path)}")
            plt.xlabel("Real (I)")
            plt.ylabel("Imaginary (Q)")
            plt.grid(True)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"  ✗ Erreur lors du traitement de {os.path.basename(h5_path)} : {e}")

if __name__ == '__main__':
    main()
