import matplotlib
matplotlib.use('Agg')
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Import metrics from metrics module
from metrics import (
    calculate_all_metrics,
    calculate_mse,
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_smape,
    print_metrics as print_metrics_detailed
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except RuntimeError as e:
            print(f"Warning: CUDA error in set_seed: {e}")
            print("Attempting to reset CUDA state...")
            try:
                torch.cuda.empty_cache()
                # Wait a bit and retry
                import time
                time.sleep(1)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except Exception as e2:
                print(f"Failed to reset CUDA state: {e2}")
                print("Continuing with CPU seed only...")


def get_device(device_name: str = "cuda", device_id: int = None) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                print(f"Warning: GPU {device_id} not available, using GPU 0")
                device_id = 0
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logger(log_dir: str, exp_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    
    if best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss


def print_metrics(metrics: Dict[str, float], logger: Optional[logging.Logger] = None):
    """
    Pretty print metrics (backward compatibility wrapper)
    Uses the detailed print function from metrics.py
    """
    print_metrics_detailed(metrics, logger)


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

