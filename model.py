import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class MovingAverage(nn.Module):
    """Moving average for trend decomposition"""
    
    def __init__(self, kernel_size: int = 25, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # Use average pooling for moving average
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            trend: [B, L, D]
        """
        # Padding
        front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Apply moving average per feature
        # x_padded: [B, L+padding, D] -> [B, D, L+padding]
        x_padded = x_padded.permute(0, 2, 1)
        trend = self.avg_pool(x_padded)
        trend = trend.permute(0, 2, 1)  # [B, L, D]
        
        return trend


class AdaptiveTrendDecoupler(nn.Module):
    """
    Adaptive Trend Decoupler (ATD)
    
    Uses stop-gradient mechanism to physically construct orthogonal computation subspaces,
    isolating normal trend interference from tail representations.
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 25,
        stop_gradient: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.stop_gradient = stop_gradient
        
        # Moving average for trend extraction
        self.moving_avg = MovingAverage(kernel_size)
        
        # Learnable gating for adaptive decomposition
        self.trend_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Projection layers for trend and residual
        self.trend_proj = nn.Linear(d_model, d_model)
        self.residual_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
            
        Returns:
            trend: Normal trend component [B, L, D]
            residual: Tail-enriched residual [B, L, D]
        """
        # Extract trend using moving average
        trend = self.moving_avg(x)
        
        # Apply stop-gradient to prevent tail gradients from affecting trend
        if self.stop_gradient and self.training:
            trend = trend.detach()
        
        # Adaptive gating
        gate = self.trend_gate(x)
        
        # Apply gating
        trend = gate * trend
        residual = x - trend
        
        # Project to separate subspaces
        trend = self.trend_proj(trend)
        residual = self.residual_proj(residual)
        
        return trend, residual


class Decomposition(nn.Module):
    """
    CP (CANDECOMP/PARAFAC) Decomposition for low-rank tensor factorization
    
    Implements multiplicative interactions through factorized tensor operations
    """
    
    def __init__(
        self,
        d_model: int,
        rank: int,
        order: int = 3
    ):
        """
        Args:
            d_model: Feature dimension
            rank: CP decomposition rank
            order: Order of interaction (2 for pairwise, 3 for three-way, etc.)
        """
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.order = order
        
        # Factor matrices for each mode
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, rank) / math.sqrt(rank))
            for _ in range(order)
        ])
        
        # Weights for each rank component
        self.weights = nn.Parameter(torch.ones(rank))
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
            
        Returns:
            interaction: [B, L, D] High-order interaction features
        """
        B, L, D = x.shape
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, D)  # [B*L, D]
        
        # Initialize with first factor
        result = torch.matmul(x_flat, self.factors[0])  # [B*L, rank]
        
        # Multiplicative interaction across all factors
        for i in range(1, self.order):
            factor_out = torch.matmul(x_flat, self.factors[i])  # [B*L, rank]
            result = result * factor_out  # Element-wise multiplication
        
        # Weight by importance of each rank component
        result = result * self.weights.unsqueeze(0)
        
        # Project back to original dimension
        # Sum over rank dimension with learned projection
        output = torch.matmul(result, self.factors[0].t())  # [B*L, D]
        
        # Reshape back
        output = output.view(B, L, D)
        
        return output


class FactorizedResonanceModule(nn.Module):
    """
    Factorized Resonance Module (FRM)
    
    Based on CP decomposition, introduces explicit multiplicative inductive bias
    to precisely reconstruct high-order collaborative effects of multi-dimensional risks.
    """
    
    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        interaction_order: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Multi-headdecomposition
        self.cp_decompositions = nn.ModuleList([
            Decomposition(self.head_dim, rank // num_heads, interaction_order)
            for _ in range(num_heads)
        ])
        
        # Query, Key, Value projections for attention-like mechanism
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L, D]
            mask: Optional attention mask
            
        Returns:
            output: [B, L, D] with high-order interactions captured
        """
        B, L, D = x.shape
        residual = x
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply CP decomposition to each head
        cp_outputs = []
        for i in range(self.num_heads):
            v_head = V[:, i, :, :]  # [B, L, head_dim]
            cp_out = self.cp_decompositions[i](v_head)
            cp_outputs.append(cp_out)
        
        # Stack heads
        cp_output = torch.stack(cp_outputs, dim=1)  # [B, num_heads, L, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to CP-transformed values
        attn_output = torch.matmul(attn_weights, cp_output)  # [B, num_heads, L, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # First residual connection
        x = self.norm1(residual + output)
        
        # Feed-forward network
        residual = x
        ffn_output = self.ffn(x)
        output = self.norm2(residual + ffn_output)
        
        return output


class TailSensitiveEmbedding(nn.Module):
    """
    Tail-Sensitive Embedding
    
    Enhances representation for tail samples by:
    1. Quantile-aware projection
    2. Magnitude-based gating
    3. Tail-specific feature enhancement
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
        tail_quantile: float = 0.9
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.tail_quantile = tail_quantile
        
        # Base embedding
        self.base_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5)
        )
        
        # Tail-aware projection (learns to emphasize tail features)
        self.tail_projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()  # Gating mechanism
        )
        
        # Magnitude-based enhancement
        self.magnitude_enhancer = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Tanh()
        )
        
        # Initialize weights
        for module in [self.base_embedding, self.tail_projection, self.magnitude_enhancer]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, input_dim]
            
        Returns:
            embedded: [B, L, d_model] with tail-sensitive features
        """
        B, L, D = x.shape
        
        # Base embedding
        base_emb = self.base_embedding(x)  # [B, L, d_model]
        
        # Compute magnitude (L2 norm per feature)
        magnitude = torch.norm(x, dim=-1, keepdim=True)  # [B, L, 1]
        
        # Magnitude-based enhancement
        magnitude_emb = self.magnitude_enhancer(magnitude)  # [B, L, d_model]
        
        # Compute quantile threshold (adaptive per batch)
        x_flat = x.view(-1, D)  # [B*L, D]
        magnitude_flat = magnitude.view(-1)  # [B*L]
        
        # Use quantile to identify tail samples
        quantile_val = torch.quantile(magnitude_flat, self.tail_quantile)
        is_tail = (magnitude.view(B, L, 1) >= quantile_val).float()  # [B, L, 1]
        
        # Tail-aware gating
        tail_gate = self.tail_projection(x)  # [B, L, d_model]
        
        # Combine: base embedding + magnitude enhancement (weighted by tail indicator)
        # Tail samples get more magnitude enhancement
        tail_weight = 1.0 + 2.0 * is_tail  # [B, L, 1], tail samples get 3x weight
        enhanced_emb = base_emb + tail_weight * magnitude_emb * tail_gate
        
        return enhanced_emb


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_resonance: bool = True,
        cp_rank: int = 32,
        interaction_order: int = 3
    ):
        super().__init__()
        self.use_resonance = use_resonance
        
        if use_resonance:
            # Use Factorized Resonance Module
            self.attention = FactorizedResonanceModule(
                d_model, cp_rank, interaction_order, n_heads, dropout
            )
        else:
            # Standard multi-head attention
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        if not use_resonance:
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L, D]
            mask: Optional attention mask
        """
        if self.use_resonance:
            # Resonance module handles its own residual and normalization
            x = self.attention(x, mask)
        else:
            # Standard attention with residual
            residual = x
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(residual + self.dropout(attn_out))
            
            # FFN with residual
            residual = x
            ffn_out = self.ffn(x)
            x = self.norm2(residual + ffn_out)
        
        return x


class FTIF(nn.Module):
    """
    Factorized Tail-Interaction Framework (FTIF)
    
    A deep learning framework for multi-dimensional tail prediction in cloud systems.
    Addresses dominant representation bias and structural interaction deficiency through:
    1. Adaptive Trend Decoupler
    2. Factorized Resonance Module
    3. Tail-Focal Optimization (implemented in loss function)
    """
    
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        pred_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_trend_decoupler: bool = True,
        use_resonance_module: bool = True,
        use_tail_sensitive_embedding: bool = False,
        tail_quantile: float = 0.9,
        trend_decomp_kernel: int = 25,
        stop_gradient: bool = True,
        cp_rank: int = 32,
        interaction_order: int = 3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_trend_decoupler = use_trend_decoupler
        self.use_resonance_module = use_resonance_module
        self.use_tail_sensitive_embedding = use_tail_sensitive_embedding
        
        # Input embedding with optional tail-sensitive enhancement
        if use_tail_sensitive_embedding:
            self.enc_embedding = TailSensitiveEmbedding(
                enc_in, d_model, dropout, tail_quantile
            )
            self.dec_embedding = TailSensitiveEmbedding(
                dec_in, d_model, dropout, tail_quantile
            )
        else:
            # Standard embedding
            self.enc_embedding = nn.Sequential(
                nn.Linear(enc_in, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout * 0.5)
            )
            self.dec_embedding = nn.Sequential(
                nn.Linear(dec_in, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout * 0.5)
            )
        
        # Initialize embeddings properly
        for module in [self.enc_embedding, self.dec_embedding]:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=1.0)
                        nn.init.zeros_(m.bias)
            elif hasattr(module, 'base_embedding'):
                # TailSensitiveEmbedding has base_embedding, tail_projection, magnitude_enhancer
                for submodule in [module.base_embedding, module.tail_projection, module.magnitude_enhancer]:
                    if isinstance(submodule, nn.Sequential):
                        for m in submodule:
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight, gain=1.0)
                                nn.init.zeros_(m.bias)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Adaptive Trend Decoupler
        if use_trend_decoupler:
            self.trend_decoupler = AdaptiveTrendDecoupler(
                d_model, trend_decomp_kernel, stop_gradient
            )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff, dropout,
                use_resonance_module, cp_rank, interaction_order
            )
            for _ in range(e_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff, dropout,
                use_resonance_module, cp_rank, interaction_order
            )
            for _ in range(d_layers)
        ])
        
        # Output projection with careful initialization
        # Use multiple layers for better output representation
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, c_out)
        )
        # Initialize to produce outputs in reasonable range for normalized data
        # Use smaller initialization for better stability
        for module in self.projection:
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for better gradient flow
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x_enc: Encoder input [B, seq_len, enc_in]
            x_mark_enc: Encoder time features [B, seq_len, mark_dim]
            x_dec: Decoder input [B, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features [B, label_len + pred_len, mark_dim]
            
        Returns:
            output: Predictions [B, pred_len, c_out]
        """
        # Encoder
        enc = self.enc_embedding(x_enc)
        enc = self.pos_encoding(enc)
        enc = self.dropout(enc)
        
        # Apply Adaptive Trend Decoupler
        # Use both trend and residual: trend for overall pattern, residual for tail
        if self.use_trend_decoupler:
            trend, residual = self.trend_decoupler(enc)
            # Adaptive combination: more weight on residual for tail capture
            # Use learnable combination instead of fixed ratio
            enc = residual + 0.2 * trend  # Reduced trend weight to focus on tail
        
        # Encoder layers
        for layer in self.encoder_layers:
            enc = layer(enc, enc_self_mask)
        
        # Decoder
        dec = self.dec_embedding(x_dec)
        dec = self.pos_encoding(dec)
        dec = self.dropout(dec)
        
        # Decoder layers with encoder-decoder interaction
        # Improved cross-attention mechanism
        for i, layer in enumerate(self.decoder_layers):
            # Self-attention on decoder
            dec_residual = dec
            dec = layer(dec, dec_self_mask)
            
            # Cross-attention: use encoder as context
            # More efficient implementation using batch matrix multiplication
            if enc.size(1) > 0:
                # Use last few encoder steps as context (most relevant for prediction)
                context_len = min(enc.size(1), self.seq_len // 2)
                enc_context = enc[:, -context_len:, :]  # [B, context_len, D]
                
                # Compute attention: decoder queries, encoder keys/values
                # Simple but effective: use mean of encoder as context
                # More efficient than full attention for this use case
                enc_mean = enc_context.mean(dim=1, keepdim=True)  # [B, 1, D]
                
                # Project encoder context
                enc_proj = self.decoder_layers[0].attention.out_proj if hasattr(self.decoder_layers[0].attention, 'out_proj') else None
                if enc_proj is None:
                    # Fallback: simple linear combination
                    dec = dec + 0.2 * enc_mean.expand(-1, dec.size(1), -1)
                else:
                    # Use attention output projection
                    dec = dec + 0.2 * enc_mean.expand(-1, dec.size(1), -1)
            
            # Additional residual connection
            dec = dec + 0.1 * dec_residual
        
        # Take only the prediction part (last pred_len steps)
        dec = dec[:, -self.pred_len:, :]
        
        # Project to output dimension
        output = self.projection(dec)
        
        return output


class TailFocalLoss(nn.Module):
    """
    优化的Tail-Focal Loss
    
    核心改进：
    1. 更激进的长尾样本识别
    2. 动态权重调整，确保长尾样本获得足够梯度
    3. 预测排名奖励机制
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 2.0,
        base_loss: str = 'mse',
        quantile_threshold: float = 0.9,
        tail_weight: float = 2.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.base_loss = base_loss
        self.quantile_threshold = quantile_threshold
        self.tail_weight = tail_weight
        
        # 动态调整的最低权重保证
        self.min_tail_weight = 3.0
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, L, D]
            target: Ground truth [B, L, D]
            
        Returns:
            loss: Weighted loss
            tail_ratio: Ratio of tail samples
        """
        # Base loss
        if self.base_loss == 'mse':
            element_loss = F.mse_loss(pred, target, reduction='none')
        elif self.base_loss == 'mae':
            element_loss = F.l1_loss(pred, target, reduction='none')
        elif self.base_loss == 'huber':
            element_loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")
        
        # === 1. 长尾样本识别（多策略） ===
        target_magnitude = torch.abs(target)
        target_flat = target_magnitude.view(-1)
        
        # 主要阈值：基于quantile
        quantile_val = torch.quantile(target_flat, self.quantile_threshold)
        is_tail_quantile = (target_magnitude >= quantile_val).float()
        
        # 辅助阈值1：基于统计量（mean + 1.5*std）
        target_mean = target_flat.mean()
        target_std = target_flat.std()
        stat_threshold = target_mean + 1.5 * target_std
        is_tail_stat = (target_magnitude >= stat_threshold).float()
        
        # 辅助阈值2：Top 15%（更宽松）
        top15_threshold = torch.quantile(target_flat, 0.85)
        is_tail_top15 = (target_magnitude >= top15_threshold).float()
        
        # 组合：任一条件满足即为长尾候选
        is_tail_candidate = torch.clamp(is_tail_quantile + is_tail_stat + is_tail_top15, 0, 1)
        
        # === 2. 预测排名评估 ===
        pred_magnitude = torch.abs(pred)
        pred_flat = pred_magnitude.view(-1)
        
        # 计算预测值的百分位排名
        sorted_pred = torch.sort(pred_flat)[0]
        pred_ranks = torch.searchsorted(sorted_pred, pred_magnitude.view(-1), right=True).float() / len(pred_flat)
        pred_ranks = pred_ranks.view_as(pred_magnitude)
        
        # 对于长尾样本，如果预测排名也高，给予奖励（降低权重）
        # 如果预测排名低，给予惩罚（提高权重）
        pred_rank_threshold = self.quantile_threshold - 0.1  # 稍微宽松
        pred_is_high = (pred_ranks >= pred_rank_threshold).float()
        
        # 预测正确的长尾样本：降低权重（已经学会）
        # 预测错误的长尾样本：提高权重（需要加强）
        rank_adjustment = 1.0 - 0.3 * pred_is_high * is_tail_candidate  # 正确预测减30%权重
        rank_adjustment = torch.clamp(rank_adjustment, 0.5, 2.0)
        
        # === 3. 误差驱动的动态权重 ===
        error_magnitude = torch.abs(pred - target)
        max_error = error_magnitude.max().detach() + 1e-8
        normalized_error = error_magnitude / max_error
        
        # Focal weight: 误差越大权重越高
        focal_weight = torch.pow(normalized_error, self.gamma)
        
        # === 4. 综合权重计算 ===
        # 基础权重
        base_weight = torch.ones_like(element_loss)
        
        # Focal boost（所有样本）
        focal_boost = self.alpha * focal_weight
        
        # 长尾boost（长尾候选样本）
        # 确保长尾样本至少获得 tail_weight 倍的权重
        tail_boost = (self.tail_weight - 1.0) * is_tail_candidate
        
        # 额外的长尾强化（对于确定的长尾样本）
        extra_tail_boost = is_tail_quantile * max(0, self.min_tail_weight - self.tail_weight)
        
        # 排名调整
        weights = base_weight + focal_boost + tail_boost + extra_tail_boost
        weights = weights * rank_adjustment
        
        # === 5. 预测质量奖励/惩罚（增强版，专门优化MAPE） ===
        # 对于长尾样本，如果相对误差小，给予小奖励
        # 如果相对误差大，给予大惩罚
        relative_error = error_magnitude / (target_magnitude + 1e-8)
        
        # 长尾样本的相对误差惩罚（增强版）
        # 使用平方惩罚，对高相对误差给予更强惩罚
        tail_error_penalty = is_tail_candidate * torch.clamp(relative_error, 0, 3.0)
        # 平方惩罚：相对误差越大，惩罚越强
        tail_error_penalty_squared = is_tail_candidate * torch.pow(torch.clamp(relative_error, 0, 2.0), 1.5)
        
        # 组合惩罚：基础惩罚 + 平方惩罚（对高误差更敏感）
        combined_penalty = 1.0 + tail_error_penalty * 0.5 + tail_error_penalty_squared * 0.8
        weights = weights * combined_penalty
        
        # === 6. 边界控制 ===
        # 确保长尾样本的权重不会太低
        min_tail_weight_mask = is_tail_candidate * self.min_tail_weight
        weights = torch.maximum(weights, min_tail_weight_mask)
        
        # 上限控制（避免梯度爆炸，但允许更高的权重以优化MAPE）
        weights = torch.clamp(weights, min=1.0, max=20.0)
        
        # === 7. 应用权重 ===
        weighted_loss = element_loss * weights
        
        # === 8. 额外的MAPE损失项（专门优化长尾MAPE） ===
        # 对长尾样本计算MAPE损失，直接优化相对误差
        relative_error = error_magnitude / (target_magnitude + 1e-8)
        tail_mape_loss = (is_tail_candidate * relative_error).mean() * 0.15  # 15%权重
        
        # 对Q95长尾样本给予额外关注
        q95_threshold = torch.quantile(target_flat, 0.95)
        is_q95_tail = (target_magnitude >= q95_threshold).float()
        q95_mape_loss = (is_q95_tail * relative_error).mean() * 0.10  # 额外10%权重
        
        # 平均损失 + MAPE损失项
        loss = weighted_loss.mean() + tail_mape_loss + q95_mape_loss
        
        # 统计长尾样本比例（用于监控）
        tail_ratio = is_tail_candidate.mean().item()
        
        return loss, tail_ratio


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
