# -*- coding: utf-8 -*-
# compare_red_results.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    input_dirs = ["output_3", "output_3_old", "output_RED_transients"]
    output_dir = "plots_RED"
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    plt.figure(figsize=(14, 7))

    csv_files = []
    for d in input_dirs:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "results_*.csv"))
            csv_files.extend([(d, f) for f in found])

    if not csv_files:
        print(f"No result CSV files found in {input_dirs}.")
        return

    for folder, file in csv_files:
        try:
            df = pd.read_csv(file)
            base_name = os.path.basename(file).replace("results_", "").replace(".csv", "")
            name = f"[{folder}] {base_name}"
            
            # Overall accuracy
            if 'correct' in df.columns:
                acc = df['correct'].mean()
                summary.append({'source_dir': folder, 'model_config': base_name, 'overall_accuracy': acc})
                
                # Plot accuracy vs window index for comparison
                if 'window_index' in df.columns:
                    acc_per_chunk = df.groupby('window_index')['correct'].mean().reset_index()
                    plt.plot(acc_per_chunk['window_index'], acc_per_chunk['correct'], label=name, marker='o', markersize=3, alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not process {file}: {e}")

    if summary:
        summary_df = pd.DataFrame(summary).sort_values(by='overall_accuracy', ascending=False)
        print("\n--- Overall Accuracy Summary (All Campaigns) ---")
        print(summary_df.to_string(index=False))
        
        summary_csv_path = os.path.join(output_dir, "overall_accuracy_summary_all.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSaved summary table to {summary_csv_path}")

    # Plot settings
    plt.title("Comprehensive Comparison: Accuracy vs. Transient Chunk (output_3, output_3_old vs RED)")
    plt.xlabel("Chunk / Window Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    
    output_plot_path = os.path.join(output_dir, "model_comparison_all_campaigns.png")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved comparison plot to {output_plot_path}")

if __name__ == "__main__":
    main()
