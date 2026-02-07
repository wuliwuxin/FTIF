import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
from typing import List
import pandas as pd

from config import (
    get_default_config,
    get_genai_config,
    get_spot_config,
    ExperimentConfig
)
from train import train_model
from evaluate import Evaluator
from utils import set_seed


def run_single_experiment(
    config: ExperimentConfig,
    target_name: str
):
    print("\n" + "="*80)
    print(f"Running Experiment: {config.exp_name}")
    print(f"Target: {target_name}")
    print("="*80 + "\n")
    
    try:
        set_seed(config.training.seed)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            print(f"Warning: CUDA error in set_seed: {e}")
            print("Clearing CUDA cache and retrying...")
            import torch
            torch.cuda.empty_cache()
            import time
            time.sleep(2)
            try:
                set_seed(config.training.seed)
            except Exception as e2:
                print(f"Failed to set seed: {e2}")
                print("Continuing anyway...")
        else:
                raise
    
    try:
        trainer, test_metrics, predictions, targets = train_model(config, target_name)
    except RuntimeError as e:
        if 'CUDA' in str(e):
            print(f"CUDA error during training: {e}")
            print("Clearing CUDA cache and skipping this experiment...")
            import torch
            torch.cuda.empty_cache()
            import time
            time.sleep(2)
            return None, None, None
        else:
            raise
    
    evaluator = Evaluator(config)
    
    save_dir = os.path.join(config.results_dir, config.exp_name, target_name)
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = evaluator.evaluate_predictions(predictions, targets, save_dir)
    
    report_path = os.path.join(save_dir, 'summary_report.txt')
    evaluator.create_summary_report(metrics, config, report_path)
    
    print(f"\n{'='*80}")
    print(f"Experiment {config.exp_name} completed successfully!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    return metrics, predictions, targets


def consolidate_results(config: ExperimentConfig):
    import glob

    print("\n" + "="*80)
    print("Consolidating Results from All Datasets")
    print("="*80 + "\n")

    all_results = []
    results_dir = config.results_dir
    metrics_files = glob.glob(os.path.join(results_dir, "**", "metrics.csv"), recursive=True)

    for metrics_file in metrics_files:
        try:
            path_parts = metrics_file.replace(os.sep, '/').split('/')
            if len(path_parts) >= 3:
                dataset_full_name = path_parts[-3]
                target_name = path_parts[-2]
                df = pd.read_csv(metrics_file)
                if len(df) > 0:
                    if '_GenTD26' in dataset_full_name or dataset_full_name.endswith('GenTD26'):
                        dataset_name = 'GenTD26'
                    elif '_Spot26' in dataset_full_name or dataset_full_name.endswith('Spot26'):
                        dataset_name = 'Spot26'
                    else:
                        parts = dataset_full_name.split('_')
                        dataset_name = parts[-1] if len(parts) > 1 else dataset_full_name
                    required_columns = {
                        'Model': 'FTIF',
                        'dataset': dataset_name,
                        'target': target_name,
                        'mse': df['mse'].iloc[0] if 'mse' in df.columns else None,
                        'mae': df['mae'].iloc[0] if 'mae' in df.columns else None,
                        'tail_mape_q95 (%)': df['tail_mape_q95 (%)'].iloc[0] if 'tail_mape_q95 (%)' in df.columns else (df['tail_mape_q95'].iloc[0] if 'tail_mape_q95' in df.columns else None),
                        'tail_capture_rate_q95': df['tail_capture_rate_q95'].iloc[0] if 'tail_capture_rate_q95' in df.columns else None
                    }
                    result_row = pd.DataFrame([required_columns])
                    all_results.append(result_row)
                    print(f"  Loaded: {dataset_full_name}/{target_name}")
        except Exception as e:
            print(f"  Warning: Failed to load {metrics_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        consolidated_df = pd.concat(all_results, ignore_index=True)
        
        column_order = ['Model', 'dataset', 'target', 'mse', 'mae',
                       'tail_mape_q95 (%)', 'tail_capture_rate_q95']
        consolidated_df = consolidated_df[column_order]
        
        consolidated_path = os.path.join(results_dir, 'consolidated_results.csv')
        consolidated_df.to_csv(consolidated_path, index=False, float_format='%.6f')
        print(f"\n{'='*80}")
        print(f"Consolidated results saved to: {consolidated_path}")
        print(f"Total experiments: {len(consolidated_df)}")
        print(f"Columns: {', '.join(column_order)}")
        print(f"All metrics saved with 6 decimal places")
        print(f"{'='*80}\n")
        
        return consolidated_df
    else:
        print("  No results found to consolidate.")
        return None


def run_genai_experiments(config: ExperimentConfig = None):
    if config is None:
        config = get_genai_config()
    
    print("\n" + "#"*80)
    print("# GenTD26 Dataset Experiments")
    print("#"*80 + "\n")
    
    results = {}
    
    for target in config.genai_targets:
        try:
            metrics, predictions, targets = run_single_experiment(config, target)
            if metrics is None:
                print(f"Skipping {target} due to training failure")
                continue
            results[target] = {
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets
            }
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import time
                time.sleep(1)
        except Exception as e:
            print(f"Error running experiment for {target}: {e}")
            import traceback
            traceback.print_exc()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import time
                time.sleep(2)
            continue
    
    return results


def run_spot_experiments(config: ExperimentConfig = None):
    if config is None:
        config = get_spot_config()
    
    print("\n" + "#"*80)
    print("# Spot26 Dataset Experiments")
    print("#"*80 + "\n")
    
    results = {}
    
    for target in config.spot_targets:
        try:
            metrics, predictions, targets = run_single_experiment(config, target)
            if metrics is None:
                print(f"Skipping {target} due to training failure")
                continue
            results[target] = {
                'metrics': metrics,
                'predictions': predictions,
                'targets': targets
            }
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import time
                time.sleep(1) 
        except Exception as e:
            print(f"Error running experiment for {target}: {e}")
            import traceback
            traceback.print_exc()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import time
                time.sleep(2)
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='FTIF: Factorized Tail-Interaction Framework'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='GenTD26',
        choices=['GenTD26', 'Spot26', 'genai', 'spot', 'both'],
        help='Experiment mode (genai=GenTD26, spot=Spot26)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Specific target variable (if not specified, run all)'
    )
    
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Custom experiment name'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    mode = args.mode
    if mode == 'genai':
        mode = 'GenTD26'
    elif mode == 'spot':
        mode = 'Spot26'
    args.mode = mode

    # Print configuration
    print("\n" + "#"*80)
    print("# FTIF: Factorized Tail-Interaction Framework")
    print("#"*80)
    print("\nCommand line arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    if args.mode == 'GenTD26':
        config = get_genai_config()
        
        if args.exp_name:
            config.exp_name = args.exp_name
        config.training.seed = args.seed
        config.training.device = args.device
        config.training.batch_size = args.batch_size
        config.training.num_epochs = args.epochs
        config.training.learning_rate = args.lr
        
        if args.target:
            config.genai_targets = [args.target]
        
        results = run_genai_experiments(config)
        
    elif args.mode == 'Spot26':
        config = get_spot_config()
        
        if args.exp_name:
            config.exp_name = args.exp_name
        config.training.seed = args.seed
        config.training.device = args.device
        config.training.batch_size = args.batch_size
        config.training.num_epochs = args.epochs
        config.training.learning_rate = args.lr
        
        if args.target:
            config.spot_targets = [args.target]
        
        results = run_spot_experiments(config)
        
    elif args.mode == 'both':
        genai_config = get_genai_config()
        genai_config.training.seed = args.seed
        genai_config.training.device = args.device
        genai_config.training.batch_size = args.batch_size
        genai_config.training.num_epochs = args.epochs
        genai_config.training.learning_rate = args.lr
        
        genai_results = run_genai_experiments(genai_config)
        
        spot_config = get_spot_config()
        spot_config.training.seed = args.seed
        spot_config.training.device = args.device
        spot_config.training.batch_size = args.batch_size
        spot_config.training.num_epochs = args.epochs
        spot_config.training.learning_rate = args.lr
        
        spot_results = run_spot_experiments(spot_config)
        
        results = {
            'GenTD26': genai_results,
            'Spot26': spot_results
        }
        
    
    consolidate_config = get_default_config()
    consolidate_config.results_dir = "./results"
    consolidate_results(consolidate_config)
    
    print("\n" + "#"*80)
    print("# All experiments completed!")
    print("#"*80 + "\n")
    
    return results


if __name__ == '__main__':
    main()

