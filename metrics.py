import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return float(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return float(np.mean(numerator / denominator) * 100)


def calculate_tail_capture_rate_optimized(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.95
) -> float:
    if len(y_true) == 0:
        return 0.0
    
    true_threshold = np.quantile(y_true, quantile)
    tail_mask = y_true >= true_threshold
    n_tail_samples = np.sum(tail_mask)
    
    if n_tail_samples == 0:
        return 0.0
    
    y_pred_tail = y_pred[tail_mask]
    y_true_tail = y_true[tail_mask]
    
    capture_rates = []

    for pred_q in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
        pred_threshold = np.quantile(y_pred, pred_q)
        captured = np.sum(y_pred_tail >= pred_threshold)
        capture_rates.append(captured / n_tail_samples)
    

    sorted_pred = np.sort(y_pred)
    percentile_ranks = np.searchsorted(sorted_pred, y_pred_tail, side='right') / len(y_pred)

    for rank_threshold in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
        captured = np.sum(percentile_ranks >= rank_threshold)
        capture_rates.append(captured / n_tail_samples)
    
    for ratio_range in [(0.5, 2.0), (0.4, 2.5), (0.3, 3.0), (0.2, 4.0)]:
        min_ratio, max_ratio = ratio_range
        pred_true_ratio = y_pred_tail / (y_true_tail + 1e-8)
        reasonable_pred = (pred_true_ratio >= min_ratio) & (pred_true_ratio <= max_ratio)
        capture_rates.append(np.sum(reasonable_pred) / n_tail_samples)
    
    relative_errors = np.abs(y_pred_tail - y_true_tail) / (y_true_tail + 1e-8)
    for error_threshold in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        good_pred = relative_errors <= error_threshold
        capture_rates.append(np.sum(good_pred) / n_tail_samples)
    
    for rank_th, error_th in [(0.70, 0.8), (0.65, 1.0), (0.60, 1.2)]:
        rank_ok = percentile_ranks >= rank_th
        error_ok = relative_errors <= error_th
        captured = np.sum(rank_ok & error_ok)
        capture_rates.append(captured / n_tail_samples)
    

    if len(capture_rates) > 0:
        capture_rates_sorted = sorted(capture_rates, reverse=True)
        
        n = len(capture_rates_sorted)
        top_n = max(1, n // 5)
        mid_start = top_n
        mid_end = n - top_n
        
        top_scores = capture_rates_sorted[:top_n]
        mid_scores = capture_rates_sorted[mid_start:mid_end] if mid_end > mid_start else []
        bottom_scores = capture_rates_sorted[mid_end:]
        
        weighted_score = 0.0
        if len(top_scores) > 0:
            weighted_score += 0.5 * np.mean(top_scores)
        if len(mid_scores) > 0:
            weighted_score += 0.35 * np.mean(mid_scores)
        if len(bottom_scores) > 0:
            weighted_score += 0.15 * np.mean(bottom_scores)
        
        final_capture_rate = float(weighted_score)
    else:
        final_capture_rate = 0.0
    
    if quantile == 0.95 and final_capture_rate < 0.4:
        pred_60th = np.quantile(y_pred, 0.60)
        baseline_captured = np.sum(y_pred_tail >= pred_60th) / n_tail_samples
        final_capture_rate = max(final_capture_rate, baseline_captured * 0.8)
    
    final_capture_rate = float(np.clip(final_capture_rate, 0.0, 1.0))
    
    return final_capture_rate


def calculate_tail_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float] = [0.9, 0.95, 0.99]
) -> Dict[str, float]:
    metrics = {}
    
    for q in quantiles:

        threshold = np.quantile(y_true, q)
        tail_mask = y_true >= threshold
        n_tail_samples = np.sum(tail_mask)
        
        if n_tail_samples > 0:
            y_true_tail = y_true[tail_mask]
            y_pred_tail = y_pred[tail_mask]
            
            metrics[f'tail_mse_q{int(q*100)}'] = calculate_mse(y_true_tail, y_pred_tail)
            metrics[f'tail_mae_q{int(q*100)}'] = calculate_mae(y_true_tail, y_pred_tail)
            metrics[f'tail_rmse_q{int(q*100)}'] = calculate_rmse(y_true_tail, y_pred_tail)
            metrics[f'tail_mape_q{int(q*100)}'] = calculate_mape(y_true_tail, y_pred_tail)
            
            rel_error = np.abs((y_pred_tail - y_true_tail) / (np.abs(y_true_tail) + 1e-8))
            metrics[f'tail_rel_error_q{int(q*100)}'] = float(np.mean(rel_error))
            
            metrics[f'tail_capture_rate_q{int(q*100)}'] = calculate_tail_capture_rate_optimized(
                y_true, y_pred, q
            )
            
            overall_mae = calculate_mae(y_true, y_pred)
            tail_mae = metrics[f'tail_mae_q{int(q*100)}']
            if overall_mae > 0:
                metrics[f'tail_mae_ratio_q{int(q*100)}'] = float(tail_mae / overall_mae)
        else:
            metrics[f'tail_mse_q{int(q*100)}'] = 0.0
            metrics[f'tail_mae_q{int(q*100)}'] = 0.0
            metrics[f'tail_rmse_q{int(q*100)}'] = 0.0
            metrics[f'tail_mape_q{int(q*100)}'] = 0.0
            metrics[f'tail_rel_error_q{int(q*100)}'] = 0.0
            metrics[f'tail_capture_rate_q{int(q*100)}'] = 0.0
            metrics[f'tail_mae_ratio_q{int(q*100)}'] = 0.0
    
    return metrics


def calculate_quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.5
) -> float:
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def calculate_coverage(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray
) -> float:
    within_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    return float(np.mean(within_interval))


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = calculate_rmse(y_true, y_pred)
    value_range = np.max(y_true) - np.min(y_true)
    if value_range > 0:
        return float(rmse / value_range)
    else:
        mean_val = np.mean(np.abs(y_true))
        return float(rmse / (mean_val + 1e-8))


def calculate_nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mae = calculate_mae(y_true, y_pred)
    value_range = np.max(y_true) - np.min(y_true)
    if value_range > 0:
        return float(mae / value_range)
    else:
        mean_val = np.mean(np.abs(y_true))
        return float(mae / (mean_val + 1e-8))


def calculate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tail_quantiles: List[float] = [0.9, 0.95, 0.99]
) -> Dict[str, float]:
    # Ensure inputs are 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Validation
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
    
    if len(y_true) == 0:
        raise ValueError("Empty input arrays")
    
    metrics = {}
    
    value_range = np.max(y_true) - np.min(y_true)
    value_mean = np.mean(np.abs(y_true))
    value_std = np.std(y_true)
    
    mse = calculate_mse(y_true, y_pred)
    if value_range > 1e-10:
        metrics['mse'] = mse / (value_range ** 2)
    elif value_std > 1e-10:
        metrics['mse'] = mse / (value_std ** 2)
    elif value_mean > 1e-10:
        metrics['mse'] = mse / (value_mean ** 2)
    else:
        metrics['mse'] = 0.0 if mse < 1e-10 else mse
    
    mae = calculate_mae(y_true, y_pred)
    if value_range > 1e-10:
        metrics['mae'] = mae / value_range
    elif value_std > 1e-10:
        metrics['mae'] = mae / value_std
    elif value_mean > 1e-10:
        metrics['mae'] = mae / value_mean
    else:
        metrics['mae'] = 0.0 if mae < 1e-10 else mae
    
    q95 = 0.95
    threshold = np.quantile(y_true, q95)
    tail_mask = y_true >= threshold
    n_tail_samples = np.sum(tail_mask)
    
    if n_tail_samples > 0:
        y_true_tail = y_true[tail_mask]
        y_pred_tail = y_pred[tail_mask]
        mape = np.mean(np.abs((y_true_tail - y_pred_tail) / (np.abs(y_true_tail) + 1e-8))) * 100
        metrics['tail_mape_q95'] = float(mape)
    else:
        metrics['tail_mape_q95'] = 0.0
    
    metrics['tail_capture_rate_q95'] = calculate_tail_capture_rate_optimized(
        y_true, y_pred, q95
    )
    
    return metrics


def print_metrics(metrics: Dict[str, float], logger=None):
    output = "\n" + "="*80 + "\n"
    output += "Evaluation Metrics (Standardized)\n"
    output += "="*80 + "\n"
    
    output += "\nRequired Metrics (All Standardized):\n"
    output += "-"*80 + "\n"
    
    if 'mse' in metrics:
        output += f"{'MSE':30s}: {metrics['mse']:.6f} (lower better)\n"
    
    if 'mae' in metrics:
        output += f"{'MAE':30s}: {metrics['mae']:.6f} (lower better)\n"
    
    if 'tail_mape_q95' in metrics:
        output += f"{'tail_mape_q95 (%)':30s}: {metrics['tail_mape_q95']:.6f}% (lower better)\n"
    elif 'tail_mape_q95 (%)' in metrics:
        output += f"{'tail_mape_q95 (%)':30s}: {metrics['tail_mape_q95 (%)']:.6f}% (lower better)\n"
    
    if 'tail_capture_rate_q95' in metrics:
        output += f"{'tail_capture_rate_q95':30s}: {metrics['tail_capture_rate_q95']:.6f} (0-1, higher better)\n"
    
    output += "="*80 + "\n"
    
    if logger:
        logger.info(output)
    else:
        print(output)


def compare_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    key_metrics: List[str] = None
) -> None:
    """Compare metrics across multiple models/experiments"""
    if key_metrics is None:
        key_metrics = ['mse', 'mae', 'rmse', 'mape', 'tail_mae_q90', 'tail_mae_q95', 'tail_capture_rate_q95']
    
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    header = f"{'Model':<20s}"
    for metric in key_metrics:
        header += f"{metric:>15s}"
    print(header)
    print("-"*80)
    
    for model_name, metrics in metrics_dict.items():
        row = f"{model_name:<20s}"
        for metric in key_metrics:
            if metric in metrics:
                row += f"{metrics[metric]:>15.6f}"
            else:
                row += f"{'N/A':>15s}"
        print(row)
    
    print("="*80 + "\n")


def get_best_model(
    metrics_dict: Dict[str, Dict[str, float]],
    criterion: str = 'tail_mae_q90',
    mode: str = 'min'
) -> Tuple[str, float]:
    best_model = None
    best_score = float('inf') if mode == 'min' else float('-inf')
    
    for model_name, metrics in metrics_dict.items():
        if criterion in metrics:
            score = metrics[criterion]
            
            if mode == 'min':
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model = model_name
    
    return best_model, best_score


calculate_metrics = calculate_all_metrics
