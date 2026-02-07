"""
Training script for FTIF model
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from config import ExperimentConfig
from model import FTIF, TailFocalLoss, count_parameters
from data_processing import get_data_processor
from utils import (
    set_seed, get_device, setup_logger, save_checkpoint,
    EarlyStopping, calculate_all_metrics
)


class Trainer:
    """Trainer class for FTIF model"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Setup
        set_seed(config.training.seed)
        device_id = getattr(config.training, 'device_id', None)
        self.device = get_device(config.training.device, device_id)
        self.logger = setup_logger(config.results_dir, config.exp_name)
        
        # Logging
        self.logger.info(f"Experiment: {config.exp_name}")
        self.logger.info(f"Description: {config.exp_description}")
        self.logger.info(f"Device: {self.device}")
        
        # Tensorboard
        if getattr(config, 'use_tensorboard', False) and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(config.results_dir, 'tensorboard', config.exp_name)
            self.writer = SummaryWriter(tb_dir)
        else:
            self.writer = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.current_target = None
        self.best_checkpoint_path = None
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='min'
        )
        
        # Logging interval
        self.log_interval = getattr(config.training, 'log_interval', 10)
        
    def prepare_data(self, target_name: str):
        """Prepare data loaders"""
        self.logger.info(f"\nPreparing data for target: {target_name}")
        
        processor = get_data_processor(self.config.data, self.config.dataset)
        
        self.train_loader, self.val_loader, self.test_loader = processor.create_dataloaders(
            target_name=target_name,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers
        )
        
        self.logger.info(f"Data preparation complete")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Val batches: {len(self.val_loader)}")
        self.logger.info(f"  Test batches: {len(self.test_loader)}")
        
        self.data_processor = processor
        self.current_target = target_name
        
        # Reset best metrics
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.best_checkpoint_path = None
        
    def build_model(self, input_dim: int, output_dim: int):
        """Build FTIF model"""
        self.logger.info("\nBuilding FTIF model...")
        
        model_config = self.config.model
        data_config = self.config.data
        
        self.model = FTIF(
            enc_in=input_dim,
            dec_in=output_dim,
            c_out=output_dim,
            seq_len=data_config.seq_len,
            label_len=data_config.label_len,
            pred_len=data_config.pred_len,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            e_layers=model_config.e_layers,
            d_layers=model_config.d_layers,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            activation=model_config.activation,
            use_trend_decoupler=model_config.use_trend_decoupler,
            use_resonance_module=model_config.use_resonance_module,
            use_tail_sensitive_embedding=model_config.use_tail_sensitive_embedding,
            tail_quantile=data_config.tail_quantile,
            trend_decomp_kernel=model_config.trend_decomp_kernel,
            stop_gradient=model_config.stop_gradient,
            cp_rank=model_config.cp_rank,
            interaction_order=model_config.interaction_order
        ).to(self.device)
        
        num_params = count_parameters(self.model)
        self.logger.info(f"Model built successfully")
        self.logger.info(f"  Parameters: {num_params:,}")
        
        # Loss function with enhanced tail focus
        if model_config.use_tail_focal:
            self.criterion = TailFocalLoss(
                alpha=model_config.tail_focal_alpha,
                gamma=model_config.tail_focal_gamma,
                base_loss=self.config.training.main_loss,
                quantile_threshold=data_config.tail_quantile,
                tail_weight=self.config.training.tail_loss_weight
            )
        else:
            if self.config.training.main_loss == 'mse':
                self.criterion = nn.MSELoss()
            elif self.config.training.main_loss == 'mae':
                self.criterion = nn.L1Loss()
            elif self.config.training.main_loss == 'huber':
                self.criterion = nn.SmoothL1Loss()
        
        # Optimizer
        if self.config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        
        # Scheduler
        if self.config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif self.config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs, eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            self.scheduler = None
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        # 确保模型在训练模式（对于MetaEFormer，需要调用其train()方法以设置MPP_update_flag）
        if hasattr(self.model, 'train') and callable(self.model.train):
            # 检查是否是MetaEFormerAdapter（有MPP_update_flag属性）
            if hasattr(self.model, 'MPP_update_flag'):
                self.model.train(True)  # 确保MPP_update_flag=True
            else:
                self.model.train()  # 标准PyTorch模型
        
        total_loss = 0.0
        total_tail_ratio = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (seq_x, seq_y, x_dec, seq_x_mark, seq_y_mark) in enumerate(pbar):
            seq_x = seq_x.to(self.device)
            seq_y = seq_y.to(self.device)
            x_dec = x_dec.to(self.device)
            seq_x_mark = seq_x_mark.to(self.device)
            seq_y_mark = seq_y_mark.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(seq_x, seq_x_mark, x_dec, seq_y_mark)
            
            # Target
            target = seq_y[:, -self.config.data.pred_len:, :]
            
            # 如果output不在计算图中（常见于模型forward异常后返回常数tensor），跳过该batch以避免backward崩溃
            if not isinstance(output, torch.Tensor) or (not output.requires_grad):
                self.logger.warning(
                    "  WARNING: Model output has no grad (requires_grad=False). "
                    "Skipping this batch (likely MetaEFormer forward fallback / constant output)."
                )
                torch.cuda.empty_cache()
                continue

            # Compute loss
            if isinstance(self.criterion, TailFocalLoss):
                loss, tail_ratio = self.criterion(output, target)
                total_tail_ratio += tail_ratio
            else:
                loss = self.criterion(output, target)
                tail_ratio = 0.0

            # 同理：如果loss不在计算图中，跳过
            if not isinstance(loss, torch.Tensor) or (not loss.requires_grad):
                self.logger.warning(
                    "  WARNING: Loss has no grad (requires_grad=False). "
                    "Skipping this batch (likely constant output / detached graph)."
                )
                torch.cuda.empty_cache()
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            
            self.optimizer.step()
            
            # Check loss validity
            loss_value = loss.item()
            if not (np.isfinite(loss_value) and loss_value < 1e6):
                self.logger.warning(f"  Abnormal loss: {loss_value}, skipping batch")
                torch.cuda.empty_cache()
                continue
            
            total_loss += loss_value
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss_value,
                'tail_ratio': tail_ratio
            })
            
            if self.writer and batch_idx % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss_value, global_step)
        
        if num_batches == 0:
            return {'loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
        
        avg_loss = total_loss / num_batches
        avg_tail_ratio = total_tail_ratio / num_batches if isinstance(self.criterion, TailFocalLoss) else 0.0
        
        if not (np.isfinite(avg_loss) and avg_loss < 1e6):
            self.logger.error(f"  Abnormal average loss: {avg_loss}")
            torch.cuda.empty_cache()
            return {'loss': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
        
        return {'loss': avg_loss, 'tail_ratio': avg_tail_ratio}
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        # 确保模型在评估模式（对于MetaEFormer，需要调用其eval()方法以设置MPP_update_flag=False）
        if hasattr(self.model, 'eval') and callable(self.model.eval):
            # 检查是否是MetaEFormerAdapter（有MPP_update_flag属性）
            if hasattr(self.model, 'MPP_update_flag'):
                self.model.eval()  # 这会设置MPP_update_flag=False
            else:
                self.model.eval()  # 标准PyTorch模型
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for seq_x, seq_y, x_dec, seq_x_mark, seq_y_mark in self.val_loader:
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                x_dec = x_dec.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)
                
                output = self.model(seq_x, seq_x_mark, x_dec, seq_y_mark)
                target = seq_y[:, -self.config.data.pred_len:, :]
                
                if isinstance(self.criterion, TailFocalLoss):
                    loss, _ = self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        predictions_flat = predictions.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Check for NaN or Inf values
        if np.isnan(predictions_flat).any() or np.isinf(predictions_flat).any():
            self.logger.warning("  WARNING: Validation predictions contain NaN or Inf, replacing with 0")
            predictions_flat = np.nan_to_num(predictions_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(targets_flat).any() or np.isinf(targets_flat).any():
            self.logger.warning("  WARNING: Validation targets contain NaN or Inf, replacing with 0")
            targets_flat = np.nan_to_num(targets_flat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Diagnostic: check for constant values
        target_std = np.std(targets_flat)
        pred_std = np.std(predictions_flat)
        target_range = np.max(targets_flat) - np.min(targets_flat)
        pred_range = np.max(predictions_flat) - np.min(predictions_flat)
        
        if target_std < 1e-6:
            self.logger.warning(f"  WARNING: Validation targets are constant (std={target_std:.2e}, range={target_range:.2e})")
        if pred_std < 1e-6:
            self.logger.warning(f"  WARNING: Validation predictions are constant (std={pred_std:.2e}, range={pred_range:.2e})")
        
        metrics = calculate_all_metrics(
            targets_flat, predictions_flat,
            tail_quantiles=self.config.eval.tail_quantiles
        )
        
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Starting training")
        self.logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            if train_metrics['loss'] == float('inf'):
                self.logger.error("Training failed, stopping...")
                break
            
            # Validate
            try:
                val_metrics = self.validate()
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    self.logger.error(f"CUDA error: {e}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
            
            # Log
            self.logger.info(f"\nEpoch {self.current_epoch}/{self.config.training.num_epochs}")
            # Use scientific notation for very small loss values
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            train_loss_str = f"{train_loss:.6e}" if train_loss < 0.0001 else f"{train_loss:.6f}"
            val_loss_str = f"{val_loss:.6e}" if val_loss < 0.0001 else f"{val_loss:.6f}"
            self.logger.info(f"  Train Loss: {train_loss_str}")
            self.logger.info(f"  Val Loss: {val_loss_str}")
            self.logger.info(f"  Val MAE: {val_metrics['mae']:.6f}")
            self.logger.info(f"  Val Tail Capture Q95: {val_metrics.get('tail_capture_rate_q95', 0):.6f}")
            # Additional diagnostic info
            if 'mse_normalized' in val_metrics:
                mse_norm = val_metrics.get('mse_normalized', 0)
                mse_str = f"{mse_norm:.6e}" if mse_norm < 0.0001 else f"{mse_norm:.6f}"
                self.logger.info(f"  Val MSE (norm): {mse_str}")
            if 'correlation' in val_metrics:
                self.logger.info(f"  Val Correlation: {val_metrics.get('correlation', 0):.6f}")
            # Diagnostic: check for constant values
            if hasattr(self, 'val_loader') and len(self.val_loader) > 0:
                # This will be computed in validate() but we can add a warning here
                pass
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], self.current_epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], self.current_epoch)
                self.writer.add_scalar('Val/TailCaptureQ95', val_metrics.get('tail_capture_rate_q95', 0), self.current_epoch)
            
            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_metrics = val_metrics.copy()
                self.logger.info("  [OK] New best model!")
            
            if self.config.training.save_best and is_best:
                target_name = self.current_target or "unknown"
                dataset_name = self.config.dataset
                filename = f"FTIF_{dataset_name}.pth"
                save_path = os.path.join(self.config.training.save_dir, filename)
                checkpoint_data = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'target_name': target_name,
                    'dataset_name': dataset_name
                }
                torch.save(checkpoint_data, save_path)
                self.best_checkpoint_path = save_path
                self.logger.info(f"  Saved to {save_path}")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"\nEarly stopping at epoch {self.current_epoch}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Training complete! Time: {total_time/60:.2f} min")
        self.logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        self.logger.info(f"{'='*50}")
        
        if self.writer:
            self.writer.close()
    
    def test(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Test model"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Testing model")
        self.logger.info(f"{'='*50}")
        
        # Load best model
        target_name = self.current_target or "unknown"
        dataset_name = self.config.dataset
        
        best_model_path = None
        if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
            best_model_path = self.best_checkpoint_path
        else:
            checkpoint_dir = self.config.training.save_dir
            candidate = os.path.join(checkpoint_dir, f"FTIF_{dataset_name}.pth")
            if os.path.isfile(candidate):
                best_model_path = candidate
        
        if best_model_path and os.path.exists(best_model_path):
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.warning("Best model not found, using current model")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for seq_x, seq_y, x_dec, seq_x_mark, seq_y_mark in tqdm(self.test_loader, desc="Testing"):
                seq_x = seq_x.to(self.device)
                seq_y = seq_y.to(self.device)
                x_dec = x_dec.to(self.device)
                seq_x_mark = seq_x_mark.to(self.device)
                seq_y_mark = seq_y_mark.to(self.device)
                
                output = self.model(seq_x, seq_x_mark, x_dec, seq_y_mark)
                target = seq_y[:, -self.config.data.pred_len:, :]
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform
        if hasattr(self, 'data_processor') and hasattr(self.data_processor, 'scalers'):
            target_name = getattr(self, 'current_target', 'default')
            if target_name in self.data_processor.scalers:
                scaler_info = self.data_processor.scalers[target_name]
                
                if isinstance(scaler_info, dict):
                    from data_processing import inverse_transform_with_tail_protection
                    
                    pred_shape = predictions.shape
                    target_shape = targets.shape
                    
                    # Reshape for inverse transform
                    predictions_flat = predictions.reshape(-1, predictions.shape[-1])
                    targets_flat = targets.reshape(-1, targets.shape[-1])
                    
                    # Apply inverse transform
                    predictions_denorm = inverse_transform_with_tail_protection(
                        predictions_flat, scaler_info
                    )
                    targets_denorm = inverse_transform_with_tail_protection(
                        targets_flat, scaler_info
                    )
                    
                    predictions = predictions_denorm.reshape(pred_shape)
                    targets = targets_denorm.reshape(target_shape)
                    
                    self.logger.info("Applied inverse normalization")
        
        # Save
        target_name = self.current_target or "unknown"
        save_path = os.path.join(self.config.results_dir, f"{self.config.exp_name}_{target_name}_predictions.npz")
        np.savez(save_path, predictions=predictions, targets=targets)
        self.logger.info(f"Saved predictions to {save_path}")
        
        # Metrics
        predictions_flat = predictions.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Diagnostic: check test data diversity (only log warnings)
        target_std = np.std(targets_flat)
        pred_std = np.std(predictions_flat)
        
        if target_std < 1e-6:
            self.logger.warning(f"  WARNING: Test targets are constant (std={target_std:.2e})")
        if pred_std < 1e-6:
            self.logger.warning(f"  WARNING: Test predictions are constant (std={pred_std:.2e})")
        
        metrics = calculate_all_metrics(
            targets_flat, predictions_flat,
            tail_quantiles=self.config.eval.tail_quantiles
        )
        
        # Log tail capture rates for diagnosis
        for q in [90, 95, 99]:
            key = f'tail_capture_rate_q{q}'
            if key in metrics:
                self.logger.info(f"  {key}: {metrics[key]:.6f}")
        
        from utils import print_metrics
        print_metrics(metrics, self.logger)
        
        return metrics, predictions, targets


def train_model(config: ExperimentConfig, target_name: str):
    """Train FTIF model for a specific target"""
    
    trainer = Trainer(config)
    trainer.prepare_data(target_name)
    
    sample_batch = next(iter(trainer.train_loader))
    input_dim = sample_batch[0].shape[-1]
    output_dim = sample_batch[1].shape[-1]
    
    trainer.build_model(input_dim, output_dim)
    trainer.train()
    test_metrics, predictions, targets = trainer.test()
    
    return trainer, test_metrics, predictions, targets
