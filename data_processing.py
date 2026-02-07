import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional
from config import DataConfig


def normalize_data_with_tail_protection(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    target_name: str
) -> tuple:
    
    train_flat = train_data.flatten()
    skewness = abs(stats.skew(train_flat))
    
    use_log = False
    shift = 0
    
    if skewness > 1.5:
        use_log = True
        min_val = train_flat.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            train_data = train_data + shift
        val_data = val_data + shift
        test_data = test_data + shift
        
        train_data = np.log1p(train_data)
        val_data = np.log1p(val_data)
        test_data = np.log1p(test_data)
        
        print(f"  Applied log transform (skewness={skewness:.2f})")
    
    clip_percentiles = (0.1, 99.9)
    
    if len(train_flat) > 1000:
        lower = np.percentile(train_flat, clip_percentiles[0])
        upper = np.percentile(train_flat, clip_percentiles[1])
        
        q95 = np.percentile(train_flat, 95)
        if upper < q95 * 1.2:
            upper = q95 * 1.5
        
        train_data = np.clip(train_data, lower, upper)
        val_data = np.clip(val_data, lower, upper)
        test_data = np.clip(test_data, lower, upper)
        
        print(f"  Clipped to [{lower:.4f}, {upper:.4f}]")
    else:
        lower = train_flat.min()
        upper = train_flat.max()
    
    scaler = RobustScaler()
    train_normalized = scaler.fit_transform(train_data)
    val_normalized = scaler.transform(val_data)
    test_normalized = scaler.transform(test_data)
    
    train_flat_norm = train_normalized.flatten()
    q25, q75 = np.percentile(train_flat_norm, [25, 75])
    iqr = q75 - q25
    
    if iqr > 0:
        if 'duration' in target_name.lower():
            multiplier = 10.0 
        else:
            multiplier = 6.0
        
        iqr_lower = q25 - multiplier * iqr
        iqr_upper = q75 + multiplier * iqr
        
        train_normalized = np.clip(train_normalized, iqr_lower, iqr_upper)
        val_normalized = np.clip(val_normalized, iqr_lower, iqr_upper)
        test_normalized = np.clip(test_normalized, iqr_lower, iqr_upper)
    
    train_max = np.abs(train_normalized).max()
    
    if train_max > 0:
        if 'duration' in target_name.lower():
            target_range = 5.0 
        else:
            target_range = 3.5
        
        if train_max > target_range:
            scale_factor = target_range / train_max
            train_normalized *= scale_factor
            val_normalized *= scale_factor
            test_normalized *= scale_factor
        else:
            scale_factor = 1.0
    else:
        scale_factor = 1.0
    
    scaler_info = {
        'scaler': scaler,
        'use_log': use_log,
        'shift': shift,
        'clip_lower': lower,
        'clip_upper': upper,
        'scale_factor': scale_factor,
        'train_max': train_max
    }
    
    print(f"  Normalization complete:")
    print(f"    - Log transform: {use_log}")
    print(f"    - Scale factor: {scale_factor:.4f}")
    print(f"    - Final range: [{train_normalized.min():.2f}, {train_normalized.max():.2f}]")
    
    return train_normalized, val_normalized, test_normalized, scaler_info


def inverse_transform_with_tail_protection(
    data: np.ndarray,
    scaler_info: dict
) -> np.ndarray:
    """
    Inverse transformation
    
    Args:
        data: Normalized data
        scaler_info: Scaler information saved during normalization
        
    Returns:
        Data in original scale
    """
    if scaler_info['scale_factor'] != 1.0:
        data = data / scaler_info['scale_factor']
    
    data_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    data_denorm = scaler_info['scaler'].inverse_transform(data_flat)
    data = data_denorm.reshape(data_shape)
    
    if scaler_info['use_log']:
        data = np.expm1(data)
        if scaler_info['shift'] > 0:
            data = data - scaler_info['shift']
    
    return data


def create_tail_aware_features(
    agg_df,
    target_name: str
) -> np.ndarray:
    """
    Create tail-aware features
    
    Core ideas:
    1. Basic statistics (mean, std, median)
    2. Tail quantiles (p90, p95, p99)
    3. Distribution shape metrics (skewness proxy, tail ratio)
    4. Time series features (rolling stats)
    
    Args:
        agg_df: Aggregated DataFrame
        target_name: Target name
        
    Returns:
        Feature array
    """
    features_list = []
    
    if 'value_mean' in agg_df.columns:
        features_list.append(agg_df['value_mean'].values)
    
    if 'value_median' in agg_df.columns:
        features_list.append(agg_df['value_median'].values)
    
    if 'value_std' in agg_df.columns:
        features_list.append(agg_df['value_std'].values)
    
    if 'duration' in target_name.lower():
        for p in ['p90', 'p95', 'p99']:
            col_name = f'value_{p}'
            if col_name in agg_df.columns:
                features_list.append(agg_df[col_name].values)
        
        # Tail ratio
        if 'value_tail_ratio' in agg_df.columns:
            features_list.append(agg_df['value_tail_ratio'].values)
    
    if 'value_cv' in agg_df.columns:
        features_list.append(agg_df['value_cv'].values)
    
    if len(features_list) > 0:
        features = np.column_stack(features_list)
    else:
        features = agg_df['value_mean'].values.reshape(-1, 1)
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if features.shape[1] > 1:
        feature_stds = np.std(features, axis=0)
        valid_features = feature_stds > 1e-8
        if np.any(valid_features):
            features = features[:, valid_features]
        else:
            features = features[:, 0:1]
    
    print(f"  Created {features.shape[1]} features for {target_name}")
    
    return features


# === Data Processor Classes ===

class TimeSeriesDataset(Dataset):
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, label_len: int, use_last_value_padding: bool = False):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.use_last_value_padding = use_last_value_padding
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def _create_time_features(self, length: int) -> np.ndarray:
        time_features = np.zeros((length, 4))
        for i in range(length):
            time_features[i, 0] = np.sin(2 * np.pi * i / 24)
            time_features[i, 1] = np.cos(2 * np.pi * i / 24)
            time_features[i, 2] = np.sin(2 * np.pi * i / 168)
            time_features[i, 3] = np.cos(2 * np.pi * i / 168)
        return time_features
    
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        x_enc = self.data[s_begin:s_end]
        
        d_begin = s_end - self.label_len
        d_end = s_end + self.pred_len
        y = self.data[d_begin:d_end]
        
        x_dec = np.zeros_like(y)
        x_dec[:self.label_len] = y[:self.label_len]
        
        if self.use_last_value_padding:
            last_value = y[self.label_len - 1:self.label_len]
            x_dec[self.label_len:] = np.tile(last_value, (self.pred_len, 1))
        
        x_mark_enc = self._create_time_features(self.seq_len)
        y_mark = self._create_time_features(self.label_len + self.pred_len)
        
        return (
            torch.FloatTensor(x_enc),
            torch.FloatTensor(y),
            torch.FloatTensor(x_dec),
            torch.FloatTensor(x_mark_enc),
            torch.FloatTensor(y_mark)
        )


class BaseDataProcessor:
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scalers = {}
    
    def create_dataloaders(
        self,
        target_name: str,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data, val_data, test_data = self._load_and_prepare_data(target_name)
        
        train_dataset = TimeSeriesDataset(
            train_data, self.config.seq_len, self.config.pred_len, self.config.label_len,
            use_last_value_padding=self.config.use_last_value_padding
        )
        val_dataset = TimeSeriesDataset(
            val_data, self.config.seq_len, self.config.pred_len, self.config.label_len,
            use_last_value_padding=self.config.use_last_value_padding
        )
        test_dataset = TimeSeriesDataset(
            test_data, self.config.seq_len, self.config.pred_len, self.config.label_len,
            use_last_value_padding=self.config.use_last_value_padding
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def _load_and_prepare_data(self, target_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data - to be implemented by subclasses"""
        raise NotImplementedError


class GenTD26DataProcessor(BaseDataProcessor):
    """Data processor for GenTD26 dataset"""
    
    def _load_and_prepare_data(self, target_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load GenTD26 data for specific target"""
        # Map target to file
        target_to_file = {
            'gpu_memory': 'pod_gpu_memory_used_bytes_anon.csv'
        }
        
        if target_name not in target_to_file:
            raise ValueError(f"Unknown GenTD26 target: {target_name}")
        
        file_path = os.path.join(self.config.genai_data_dir, target_to_file[target_name])
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if 'timestamp_anon' in df.columns and 'value' in df.columns:
            df['time_window'] = (df['timestamp_anon'] / self.config.aggregation_window).astype(int)
            agg_df = df.groupby('time_window')['value'].agg(['mean', 'std', 'median']).reset_index()
            agg_df = agg_df.fillna(0)
            
            features = agg_df[['mean', 'std', 'median']].values
        else:
            raise ValueError(f"Expected columns 'timestamp_anon' and 'value' in {file_path}")
        
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        train_data = features[:train_end]
        val_data = features[train_end:val_end]
        test_data = features[val_end:]
        
        if self.config.normalize:
            train_data, val_data, test_data, scaler_info = normalize_data_with_tail_protection(
                train_data, val_data, test_data, target_name
            )
            self.scalers[target_name] = scaler_info
        
        return train_data, val_data, test_data


class SpotDataProcessor(BaseDataProcessor):
    """Enhanced data processor for Spot26 dataset with feature engineering"""
    
    def _load_and_prepare_data(self, target_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Spot26 data for specific target with enhanced features"""
        if target_name != 'duration':
            raise ValueError(f"Unknown Spot26 target: {target_name}")
        
        file_path = os.path.join(self.config.spot_data_dir, 'job_info_df.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        if 'duration' not in df.columns:
            raise ValueError(f"Expected column 'duration' in {file_path}")
        
        print(f"  Loading Spot26 dataset with enhanced feature engineering...")
        print(f"  Original columns: {df.columns.tolist()}")
        
        feature_list = []
        
        if 'duration' in df.columns:
            duration = df['duration'].values.astype(float)
            feature_list.append(duration.reshape(-1, 1))
            
            duration_log = np.log1p(duration)
            feature_list.append(duration_log.reshape(-1, 1))
            
            window_size = min(100, len(duration) // 10)
            if window_size > 1:
                duration_rolling_mean = pd.Series(duration).rolling(window=window_size, min_periods=1).mean().values
                duration_rolling_std = pd.Series(duration).rolling(window=window_size, min_periods=1).std().fillna(0).values
                duration_rolling_max = pd.Series(duration).rolling(window=window_size, min_periods=1).max().values
                feature_list.append(duration_rolling_mean.reshape(-1, 1))
                feature_list.append(duration_rolling_std.reshape(-1, 1))
                feature_list.append(duration_rolling_max.reshape(-1, 1))
        
        if 'cpu_request' in df.columns:
            cpu_request = df['cpu_request'].values.astype(float)
            feature_list.append(cpu_request.reshape(-1, 1))
            if 'gpu_request' in df.columns:
                gpu_request = df['gpu_request'].values.astype(float)
                cpu_gpu_ratio = np.divide(cpu_request, gpu_request + 1e-8)
                feature_list.append(cpu_gpu_ratio.reshape(-1, 1))
        
        if 'gpu_request' in df.columns:
            gpu_request = df['gpu_request'].values.astype(float)
            feature_list.append(gpu_request.reshape(-1, 1))
        
        if 'worker_num' in df.columns:
            worker_num = df['worker_num'].values.astype(float)
            feature_list.append(worker_num.reshape(-1, 1))
            if 'gpu_request' in df.columns:
                worker_gpu_ratio = np.divide(worker_num, gpu_request + 1e-8)
                feature_list.append(worker_gpu_ratio.reshape(-1, 1))
        
        if 'submit_time' in df.columns:
            submit_time = df['submit_time'].values.astype(float)
            submit_time_norm = (submit_time - submit_time.min()) / (submit_time.max() - submit_time.min() + 1e-8)
            feature_list.append(submit_time_norm.reshape(-1, 1))
            
            time_sin = np.sin(2 * np.pi * submit_time_norm)
            time_cos = np.cos(2 * np.pi * submit_time_norm)
            feature_list.append(time_sin.reshape(-1, 1))
            feature_list.append(time_cos.reshape(-1, 1))
        
        if 'gpu_model' in df.columns:
            gpu_models = df['gpu_model'].astype(str)
            top_models = gpu_models.value_counts().head(10).index.tolist()
            for model in top_models:
                model_indicator = (gpu_models == model).astype(float).values
                feature_list.append(model_indicator.reshape(-1, 1))
        
        if 'organization' in df.columns:
            organizations = df['organization'].astype(str)
            top_orgs = organizations.value_counts().head(10).index.tolist()
            for org in top_orgs:
                org_indicator = (organizations == org).astype(float).values
                feature_list.append(org_indicator.reshape(-1, 1))
        
        if 'job_type' in df.columns:
            job_type_encoded = (df['job_type'] == 'Spot').astype(float).values
            feature_list.append(job_type_encoded.reshape(-1, 1))
        
        if 'duration' in df.columns and 'gpu_request' in df.columns:
            duration_gpu_interaction = duration * gpu_request
            feature_list.append(duration_gpu_interaction.reshape(-1, 1))
        
        if 'duration' in df.columns and 'worker_num' in df.columns:
            duration_worker_interaction = duration * worker_num
            feature_list.append(duration_worker_interaction.reshape(-1, 1))
        
        if len(feature_list) > 0:
            features = np.hstack(feature_list)
        else:
            features = df[['duration']].values
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        feature_stds = np.std(features, axis=0)
        valid_features = feature_stds > 1e-8
        if np.any(valid_features):
            features = features[:, valid_features]
            print(f"  Created {features.shape[1]} features (removed {np.sum(~valid_features)} constant features)")
        else:
            features = features[:, 0:1] if features.shape[1] > 0 else df[['duration']].values
            print(f"  Warning: All features were constant, using duration only")
        
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        
        train_data = features[:train_end]
        val_data = features[train_end:val_end]
        test_data = features[val_end:]
        
        print(f"  Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        if self.config.normalize:
            train_data, val_data, test_data, scaler_info = normalize_data_with_tail_protection(
                train_data, val_data, test_data, target_name
            )
            self.scalers[target_name] = scaler_info
        
        return train_data, val_data, test_data


def get_data_processor(data_config: DataConfig, dataset: str):
    if dataset in ('GenTD26', 'genai'):
        return GenTD26DataProcessor(data_config)
    if dataset in ('Spot26', 'spot'):
        return SpotDataProcessor(data_config)
    raise ValueError(f"Unknown dataset: {dataset}. Must be 'GenTD26', 'Spot26', 'genai', or 'spot'")
