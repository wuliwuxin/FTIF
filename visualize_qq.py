"""
Usage:
  python visualize_qq.py --npz ./results/FTIF_Spot_duration_predictions.npz
  python visualize_qq.py --exp_name FTIF_Spot --target duration [--results_dir ./results]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def plot_qq(predictions: np.ndarray, targets: np.ndarray, save_path: str):
    """Draw Q-Q plot: Ground Truth vs Predictions."""
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    n = min(len(targets), len(predictions))
    quantiles = np.linspace(0.01, 0.99, n)
    sorted_targets = np.sort(targets)
    sorted_predictions = np.sort(predictions)
    theoretical_quantiles = stats.norm.ppf(quantiles)
    ax.scatter(theoretical_quantiles, sorted_targets,
               alpha=0.7, s=40, label='Ground Truth', color='steelblue', marker='o',
               edgecolors='navy', linewidths=0.8)
    ax.scatter(theoretical_quantiles, sorted_predictions,
               alpha=0.7, s=40, label='Predictions', color='coral', marker='s',
               edgecolors='darkred', linewidths=0.8)
    ax.set_xlabel('Theoretical Quantiles (Normal Distribution)', fontsize=18)
    ax.set_ylabel('Sample Quantiles', fontsize=18)
    ax.set_title('Q-Q Plot: Ground Truth vs Predictions', fontsize=20, fontweight='bold')
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Q-Q plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Draw Q-Q plot from saved predictions (run after training).'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--npz',
        type=str,
        help='Path to predictions .npz file (e.g. results/FTIF_Spot_duration_predictions.npz)'
    )
    group.add_argument(
        '--exp_target',
        nargs=2,
        metavar=('EXP_NAME', 'TARGET'),
        help='Experiment name and target (e.g. FTIF_Spot duration)'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Results directory when using --exp_target (default: ./results)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output image path (default: results_dir/exp_name/target/exp_name_qq_plot.png)'
    )
    args = parser.parse_args()

    if args.npz:
        npz_path = os.path.abspath(args.npz)
        if not os.path.isfile(npz_path):
            print(f"Error: file not found: {npz_path}")
            return 1
        data = np.load(npz_path)
        predictions = data['predictions'].reshape(-1)
        targets = data['targets'].reshape(-1)
        # Infer exp_name and target from filename: .../FTIF_Spot_duration_predictions.npz
        basename = os.path.basename(npz_path)
        if basename.endswith('_predictions.npz'):
            parts = basename[:-len('_predictions.npz')].split('_')
            if len(parts) >= 2:
                target = parts[-1]
                exp_name = '_'.join(parts[:-1])
            else:
                exp_name, target = 'FTIF', 'default'
        else:
            exp_name, target = 'FTIF', 'default'
        results_dir = os.path.dirname(npz_path)
    else:
        exp_name, target = args.exp_target
        results_dir = os.path.abspath(args.results_dir)
        npz_path = os.path.join(results_dir, f"{exp_name}_{target}_predictions.npz")
        if not os.path.isfile(npz_path):
            print(f"Error: file not found: {npz_path}")
            return 1
        data = np.load(npz_path)
        predictions = data['predictions'].reshape(-1)
        targets = data['targets'].reshape(-1)

    if args.out:
        save_path = os.path.abspath(args.out)
    else:
        save_dir = os.path.join(results_dir, exp_name, target)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{exp_name}_qq_plot.png")

    plot_qq(predictions, targets, save_path)
    return 0


if __name__ == '__main__':
    exit(main())
