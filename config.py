import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DataConfig:
    genai_data_dir: str = "./datasets/GenTD26"
    spot_data_dir: str = "./datasets/Spot26"
    
    genai_files: List[str] = field(default_factory=lambda: [
        "pod_gpu_memory_used_bytes_anon.csv", 
    ])
    
    spot_files: List[str] = field(default_factory=lambda: [
        "job_info_df.csv",
        "node_info_df.csv"
    ])
    
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    
    normalize: bool = True
    aggregation_window: int = 60
    use_last_value_padding: bool = True
    
    tail_quantile: float = 0.9


@dataclass
class ModelConfig:
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 2
    dropout: float = 0.1
    
    trend_decomp_kernel: int = 25
    stop_gradient: bool = True
    
    cp_rank: int = 32
    interaction_order: int = 3
    resonance_heads: int = 4
    
    tail_focal_alpha: float = 3.0
    tail_focal_gamma: float = 2.5
    quantile_aware_weight: bool = True
    
    use_trend_decoupler: bool = True
    use_resonance_module: bool = True
    use_tail_focal: bool = True
    use_tail_sensitive_embedding: bool = True
    
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    
    scheduler: str = "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    patience: int = 15
    min_delta: float = 1e-6
    
    main_loss: str = "huber"
    tail_loss_weight: float = 3.0
    
    optimizer: str = "adamw"
    grad_clip: float = 0.5
    
    save_dir: str = "./checkpoints"
    save_best: bool = True
    save_interval: int = 5
    
    device: str = "cuda"
    device_id: int = None
    num_workers: int = 4
    seed: int = 42


@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "rmse", "mape", "smape"
    ])
    
    tail_quantiles: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.99])
    
    plot_predictions: bool = True
    plot_attention: bool = True
    plot_dir: str = "./results/plots"


@dataclass
class ExperimentConfig:
    exp_name: str = "FTIF_experiment"
    exp_description: str = "Factorized Tail-Interaction Framework for Multi-Dimensional Tail Prediction"
    
    dataset: str = "GenTD26"
    
    genai_targets: List[str] = field(default_factory=lambda: [
        "gpu_memory",
    ])
    
    spot_targets: List[str] = field(default_factory=lambda: [
        "duration",
    ])
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    results_dir: str = "./results"
    
    log_interval: int = 10
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = "FTIF"
    
    deterministic: bool = True
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.training.save_dir, exist_ok=True)


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


def get_genai_config() -> ExperimentConfig:
    config = ExperimentConfig(
        exp_name="FTIF_GenTD26",
        dataset="genai",
    )
    return config


def get_spot_config() -> ExperimentConfig:
    config = ExperimentConfig(
        exp_name="FTIF_Spot26",
        dataset="spot",
    )
    
    config.model.tail_focal_alpha = 4.0
    config.model.tail_focal_gamma = 3.0
    config.model.cp_rank = 48
    config.model.interaction_order = 4
    config.model.d_model = 512
    config.model.e_layers = 4
    config.model.d_layers = 3
    
    config.training.num_epochs = 100
    config.training.learning_rate = 1.5e-5
    config.training.tail_loss_weight = 4.0
    config.training.patience = 40
    config.training.batch_size = 128
    
    config.data.seq_len = 128
    config.data.pred_len = 24
    config.data.label_len = 48
    config.data.tail_quantile = 0.9
    
    print("  Spot26 dataset configuration optimized for tail prediction:")
    print(f"    - Tail focal alpha: {config.model.tail_focal_alpha}")
    print(f"    - Tail focal gamma: {config.model.tail_focal_gamma}")
    print(f"    - Tail loss weight: {config.training.tail_loss_weight}")
    print(f"    - CP rank: {config.model.cp_rank}")
    print(f"    - Interaction order: {config.model.interaction_order}")
    print(f"    - Sequence length: {config.data.seq_len}")
    
    return config