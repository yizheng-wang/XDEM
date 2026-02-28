import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '../../result/crack2/crack_XDEM/different_nn_architectures/'


def find_results_csvs():
    """Find result CSV files saved by NN architecture sweeps."""
    pattern = os.path.join(OUTPUT_DIR, 'nn_architecture_results_tau_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No results found matching: {pattern}')
    else:
        print(f'Found {len(files)} results CSV(s):')
        for f in files:
            print(f'  - {f}')
    return files


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize columns if needed
    # Expecting: depth, width, best_k2_j, best_error, best_epoch, fem_value, K2_true
    required_cols = {'depth', 'width', 'best_error'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {csv_path}: {missing}')
    return df


def plot_width_vs_error_by_depth(df: pd.DataFrame, tau_label: str):
    depths = sorted(df['depth'].unique())
    widths = sorted(df['width'].unique())

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'teal']
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P']

    for i, depth in enumerate(depths):
        slice_df = df[df['depth'] == depth].sort_values('width')
        plt.plot(slice_df['width'].values,
                 slice_df['best_error'].values,
                 color=colors[i % len(colors)],
                 marker=markers[i % len(markers)],
                 linewidth=2,
                 markersize=6,
                 label=f'Depth = {depth}')

    plt.xlabel('Network Width', fontsize=20)
    plt.ylabel('Error vs FEM', fontsize=20)
    plt.title(f'NN Architecture Comparison (tau = {tau_label})', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=20)
    # 坐标轴刻度字体
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.yscale('log')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'nn_architecture_comparison_tau_{tau_label}_from_saved.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_heatmap(df: pd.DataFrame, tau_label: str):
    pivot = df.pivot_table(index='depth', columns='width', values='best_error', aggfunc='min')
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, label='Error vs FEM (log)')
    plt.clim(vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, fontsize=16)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index, fontsize=16)
    plt.xlabel('Width', fontsize=20)
    plt.ylabel('Depth', fontsize=20)
    plt.title(f'Error Heatmap (tau = {tau_label})', fontsize=20)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'nn_architecture_heatmap_tau_{tau_label}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def summarize_best(df: pd.DataFrame, tau_label: str):
    best_row = df.loc[df['best_error'].idxmin()]
    print('\nBest configuration:')
    print(f"  tau={tau_label}, depth={int(best_row['depth'])}, width={int(best_row['width'])}, "
          f"error={best_row['best_error']:.4f}, best_epoch={int(best_row['best_epoch']) if 'best_epoch' in df.columns and not np.isnan(best_row['best_epoch']) else 'N/A'}")


def main():
    csv_files = find_results_csvs()
    if not csv_files:
        return

    for csv_path in csv_files:
        # tau label from filename
        base = os.path.basename(csv_path)
        # expects nn_architecture_results_tau_{tau}.csv
        try:
            tau_label = base.split('nn_architecture_results_tau_')[1].split('.csv')[0]
        except Exception:
            tau_label = 'unknown'
        df = load_results(csv_path)
        plot_width_vs_error_by_depth(df, tau_label)
        plot_heatmap(df, tau_label)
        summarize_best(df, tau_label)

    print('\nAll plots generated from saved results.')


if __name__ == '__main__':
    main()
