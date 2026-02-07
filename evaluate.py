import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch
from scipy import stats
import sys
import time

from config import ExperimentConfig
from utils import calculate_all_metrics, print_metrics


def safe_visualize(func, *args, max_time=300, step_name="visualization", max_samples=200000, **kwargs):
    if len(args) > 0 and hasattr(args[0], '__len__'):
        data_size = len(args[0])
        if data_size > max_samples:
            print(f"    ⚠ Skipping {step_name} (data too large: {data_size:,} samples, limit: {max_samples:,})")
            return False
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        if elapsed > 60:
            print(f"    ⚠ {step_name} took {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ⚠ Error in {step_name} after {elapsed:.1f}s: {str(e)[:100]}, skipping...")
        return False


class Evaluator:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str
    ) -> Dict[str, float]:
        os.makedirs(save_dir, exist_ok=True)
        
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        metrics = calculate_all_metrics(
            target_flat,
            pred_flat,
            tail_quantiles=self.config.eval.tail_quantiles
        )
        
        print_metrics(metrics)
        
        ordered_metrics = {}
        
        if 'mse' in metrics:
            ordered_metrics['mse'] = metrics['mse']
        if 'mae' in metrics:
            ordered_metrics['mae'] = metrics['mae']
        if 'tail_mape_q95' in metrics:
            ordered_metrics['tail_mape_q95 (%)'] = metrics['tail_mape_q95']
        if 'tail_capture_rate_q95' in metrics:
            ordered_metrics['tail_capture_rate_q95'] = metrics['tail_capture_rate_q95']
        
        metrics_df = pd.DataFrame([ordered_metrics])
        metrics_path = os.path.join(save_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
        print(f"\nMetrics saved to {metrics_path} (standardized metrics, 6 decimal places)")
        
        return metrics
    
    def _analyze_tail_behavior(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        metrics = {}
        
        for q in self.config.eval.tail_quantiles:
            threshold = np.quantile(targets, q)
            tail_mask = targets >= threshold
            
            if np.sum(tail_mask) > 0:
                tail_pred = predictions[tail_mask]
                tail_true = targets[tail_mask]
                
                rel_error = np.abs((tail_pred - tail_true) / (tail_true + 1e-8))
                metrics[f'tail_relative_error_q{int(q*100)}'] = rel_error.mean()
                
                pred_threshold = np.quantile(predictions, q)
                predicted_as_tail = predictions[tail_mask] >= pred_threshold
                capture_rate = predicted_as_tail.sum() / len(tail_mask[tail_mask])
                metrics[f'tail_capture_rate_q{int(q*100)}'] = capture_rate
        
        return metrics
    
    def visualize_results(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str,
        exp_name: str
    ):
        print("\nGenerating Q-Q plot...")
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        os.makedirs(save_dir, exist_ok=True)
        qq_path = os.path.join(save_dir, f'{exp_name}_qq_plot.png')
        try:
            self._plot_qq(pred_flat, target_flat, qq_path)
            print("Q-Q plot saved.")
        except Exception as e:
            print(f"Error generating Q-Q plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_qq(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: str
    ):
        plt.rcParams.update({'font.size': 16})
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        n = min(len(targets), len(predictions))
        quantiles = np.linspace(0.01, 0.99, n)
        
        sorted_targets = np.sort(targets)
        sorted_predictions = np.sort(predictions)
        
        from scipy.stats import norm
        theoretical_quantiles = norm.ppf(quantiles)
        
        ax.scatter(theoretical_quantiles, sorted_targets, alpha=0.7, s=40, label='Ground Truth', color='steelblue', marker='o', edgecolors='navy', linewidths=0.8)
        
        ax.scatter(theoretical_quantiles, sorted_predictions, alpha=0.7, s=40, label='Predictions', color='coral', marker='s', edgecolors='darkred', linewidths=0.8)
        
        ax.set_xlabel('Theoretical Quantiles (Normal Distribution)', fontsize=18)
        ax.set_ylabel('Sample Quantiles', fontsize=18)
        ax.set_title('Q-Q Plot: Ground Truth vs Predictions', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        data_save_path = save_path.replace('.png', '_data.npz')
        np.savz(
            data_save_path,
            theoretical_quantiles=theoretical_quantiles,
            sorted_targets=sorted_targets,
            sorted_predictions=sorted_predictions,
            quantiles=quantiles,
            n_samples=n
        )
        
        print(f"  Saved Q-Q plot to {save_path}")
        print(f"  Saved Q-Q plot data to {data_save_path}")
    
    def create_summary_report(
        self,
        metrics: Dict[str, float],
        config: ExperimentConfig,
        save_path: str
    ):
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FTIF Experiment Summary Report\n")
            f.write("="*80 + "\n\n")
            
            f.write("Experiment Configuration\n")
            f.write("-"*80 + "\n")
            f.write(f"Experiment Name: {config.exp_name}\n")
            f.write(f"Description: {config.exp_description}\n")
            f.write(f"Dataset: {config.dataset}\n")
            f.write(f"Sequence Length: {config.data.seq_len}\n")
            f.write(f"Prediction Length: {config.data.pred_len}\n")
            f.write("\n")
            
            f.write("Model Configuration\n")
            f.write("-"*80 + "\n")
            f.write(f"Model Dimension: {config.model.d_model}\n")
            f.write(f"Encoder Layers: {config.model.e_layers}\n")
            f.write(f"Decoder Layers: {config.model.d_layers}\n")
            f.write(f"Attention Heads: {config.model.n_heads}\n")
            f.write(f"Use Trend Decoupler: {config.model.use_trend_decoupler}\n")
            f.write(f"Use Resonance Module: {config.model.use_resonance_module}\n")
            f.write(f"Use Tail-Focal Loss: {config.model.use_tail_focal}\n")
            f.write(f"CP Rank: {config.model.cp_rank}\n")
            f.write(f"Interaction Order: {config.model.interaction_order}\n")
            f.write("\n")
            
            f.write("Training Configuration\n")
            f.write("-"*80 + "\n")
            f.write(f"Batch Size: {config.training.batch_size}\n")
            f.write(f"Learning Rate: {config.training.learning_rate}\n")
            f.write(f"Optimizer: {config.training.optimizer}\n")
            f.write(f"Scheduler: {config.training.scheduler}\n")
            f.write(f"Epochs: {config.training.num_epochs}\n")
            f.write("\n")
            
            f.write("Evaluation Results\n")
            f.write("-"*80 + "\n")
            
            f.write("\nCore Standardized Metrics:\n")
            core_keys = ['mse', 'mae', 'nmae', 'nrmse']
            for key in core_keys:
                if key in metrics:
                    f.write(f"  {key.upper():20s}: {metrics[key]:.6f} (lower better)\n")
            
            f.write("\nOther Standardized Metrics:\n")
            other_std_keys = ['mape', 'smape', 'correlation']
            for key in other_std_keys:
                if key in metrics:
                    if key in ['mape', 'smape']:
                        f.write(f"  {key.upper():20s}: {metrics[key]:.6f}%\n")
                    else:
                        f.write(f"  {key.upper():20s}: {metrics[key]:.6f} (higher better)\n")
            
            if 'mean_erro' in metrics:
                f.write("\nStandardized Error Statistics:\n")
                f.write(f"  Mean Error: {metrics['mean_error']:.6f}\n")
                f.write(f"  Std Error:  {metrics['std_error']:.6f}\n")
                f.write(f"  Max Error:  {metrics['max_error']:.6f}\n")
            
            f.write("\nTail Metrics (All Together):\n")
            tail_keys = [k for k in sorted(metrics.keys()) if k.startswith('tail_')]
            for key in tail_keys:
                if 'capture_rate' in key:
                    f.write(f"  {key:35s}: {metrics[key]:.6f} (0-1, higher better)\n")
                elif 'rel_error' in key or 'relative_error' in key:
                    f.write(f"  {key:35s}: {metrics[key]:.6f} (0-1, lower better)\n")
                elif 'mape' in key:
                    f.write(f"  {key:35s}: {metrics[key]:.6f}% (lower better)\n")
                elif 'nrmse' in key:
                    f.write(f"  {key:35s}: {metrics[key]:.6f} (lower better)\n")
                elif 'ratio' in key:
                    f.write(f"  {key:35s}: {metrics[key]:.6f} (ratio, lower better)\n")
                else:
                    f.write(f"  {key:35s}: {metrics[key]:.6f}\n")
            
            f.write("\nAbsolute Metrics (Reference):\n")
            absolute_keys = ['mse', 'mae', 'rmse', 'mean_error', 'std_error', 'max_error']
            for key in absolute_keys:
                if key in metrics:
                    f.write(f"  {key.upper():20s}: {metrics[key]:.6f}\n")
            
            if 'mean_erro6' in metrics:
                f.write("\nError Statistics (Standardized):\n")
                f.write(f"  Mean Error (norm): {metrics['mean']:.6f}\n")
                f.write(f"  Std Error (norm):  {metrics['std_error']:.6f}\n")
                f.write(f"  Max Error (norm):  {metrics['max_error']:.6f}\n")
            
            f.write("\nTail Metrics:\n")
            quantiles = [90, 95, 99]
            for q in quantiles:
                f.write(f"\n  Quantile Q{q}:\n")
                tail_keys = [k for k in metrics.keys() if f'q{q}' in k and k.startswith('tail_')]
                for key in sorted(tail_keys):
                    if 'capture_rate' in key:
                        f.write(f"    {key:35s}: {metrics[key]:.6f} (0-1, higher better)\n")
                    elif 'rel_error' in key or 'relative_error' in key:
                        f.write(f"    {key:35s}: {metrics[key]:.6f} (0-1, lower better)\n")
                    elif 'mape' in key:
                        f.write(f"    {key:35s}: {metrics[key]:.6f}% (lower better)\n")
                    elif 'nrmse' in key:
                        f.write(f"    {key:35s}: {metrics[key]:.6f} (lower better)\n")
                    elif 'ratio' in key:
                        f.write(f"    {key:35s}: {metrics[key]:.6f} (ratio, lower better)\n")
                    else:
                        f.write(f"    {key:35s}: {metrics[key]:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nSummary report saved to {save_path}")
