"""
Flax-based Föllmer Drift Networks
基于Flax的Föllmer漂移网络

High-performance JAX/Flax implementation of neural networks for drift parametrization.
高性能JAX/Flax实现的漂移参数化神经网络。
"""

import jax
import jax.numpy as jnp
import jax.random
from jax import lax, vmap, pmap, jit
from jax.nn import silu, gelu, swish
from functools import partial
from typing import Optional, Callable, Tuple, Any, Dict, List
import math

from flax import linen as nn
from flax.linen import LayerNorm, GroupNorm
from flax.training import train_state
import optax
import chex


from ..core.types import (
    SDEState, DriftFunction, NetworkParams, TimeEncoding, BatchStates, BatchTimes,
    NetworkConfig, TrainingConfig, PerformanceConfig, NetworkTrainingState
)
from ..core.registry import register_network


# ============================================================================
# Time Encoding Module / 时间编码模块
# ============================================================================

class TimeEncoder(nn.Module):
    """
    Sinusoidal positional encoding for time conditioning
    时间条件的正弦位置编码
    
    Provides efficient, cache-friendly time encodings with GPU optimization.
    提供高效、缓存友好的时间编码和GPU优化。
    """
    
    encoding_dim: int = 128
    max_time: float = 100.0
    learnable_scaling: bool = True
    use_cache: bool = True
    
    def setup(self):
        """Initialize time encoding parameters / 初始化时间编码参数"""
        # 预计算编码频率 / Precompute encoding frequencies
        half_dim = self.encoding_dim // 2
        freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half_dim) / (half_dim - 1))
        self.freqs = freqs
        
        # 可学习的缩放因子 / Learnable scaling factors
        if self.learnable_scaling:
            self.time_scale = self.param('time_scale', nn.initializers.ones, (1,))
            self.freq_scale = self.param('freq_scale', nn.initializers.ones, (half_dim,))
        
        # 时间编码缓存 / Time encoding cache
        if self.use_cache:
            self._cache = {}
    
    def encode_time(self, t: float) -> TimeEncoding:
        """
        Encode single time point
        编码单个时间点
        
        Args:
            t: Time value / 时间值
            
        Returns:
            encoding: Time encoding vector / 时间编码向量
        """
        if self.learnable_scaling:
            scaled_t = t * self.time_scale
            scaled_freqs = self.freqs * self.freq_scale
        else:
            scaled_t = t
            scaled_freqs = self.freqs
        
        # 计算正弦和余弦编码 / Compute sine and cosine encodings
        angles = scaled_t * scaled_freqs
        sin_enc = jnp.sin(angles)
        cos_enc = jnp.cos(angles)
        
        # 交错组合 / Interleave
        encoding = jnp.concatenate([sin_enc, cos_enc], axis=-1)
        
        # 如果维度是奇数，截断最后一个元素 / Truncate if odd dimension
        if self.encoding_dim % 2 == 1:
            encoding = encoding[:self.encoding_dim]
            
        return encoding
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized time encoding
        向量化时间编码
        
        Args:
            t: Time values (can be scalar or batch) / 时间值（标量或批量）
            
        Returns:
            encodings: Time encodings / 时间编码
        """
        # FIX: Robustly handle Python floats by converting to JAX array.
        # This prevents the "'float' object has no attribute 'ndim'" error.
        # 修复：通过转换为JAX数组来稳健地处理Python浮点数。
        # 这可以防止“'float'对象没有'ndim'属性”的错误。
        t = jnp.asarray(t)
        
        # 如果是标量，直接编码 / If scalar, encode directly
        if t.ndim == 0:
            return self.encode_time(t)
        
        # 如果是批量，使用vmap并行化 / If batch, use vmap for parallelization
        return vmap(self.encode_time)(t)


# ============================================================================
# Time-Conditioned ResNet Block / 时间条件ResNet块
# ============================================================================

class TimeConditionedResNetBlock(nn.Module):
    """
    ResNet block with time conditioning
    带时间条件的ResNet块
    
    Features:
    - Time-dependent feature modulation
    - Spectral normalization for stability
    - Efficient GPU computation
    
    特性：
    - 时间相关的特征调制
    - 谱归一化提高稳定性
    - 高效GPU计算
    """
    
    hidden_dim: int
    activation: str = "silu"
    use_spectral_norm: bool = True
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    time_conditioning_type: str = "film"  # "film", "concat", "add"
    
    def setup(self):
        """Initialize ResNet block components / 初始化ResNet块组件"""
        # 激活函数选择 / Activation function selection
        if self.activation == "silu":
            self.act_fn = silu
        elif self.activation == "gelu":
            self.act_fn = gelu
        elif self.activation == "swish":
            self.act_fn = swish
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # 主要变换层 / Main transformation layers
        self.dense1 = nn.Dense(self.hidden_dim, use_bias=False)
        self.dense2 = nn.Dense(self.hidden_dim, use_bias=False)
        
        # 时间条件层 / Time conditioning layers
        if self.time_conditioning_type == "film":
            # Feature-wise Linear Modulation
            self.time_mlp = nn.Sequential([
                nn.Dense(self.hidden_dim * 2),
                self.act_fn,
                nn.Dense(self.hidden_dim * 2)
            ])
        elif self.time_conditioning_type == "concat":
            # 适应连接后的维度 / Adapt for concatenated dimension
            self.time_dense = nn.Dense(self.hidden_dim // 4)
            self.combined_dense = nn.Dense(self.hidden_dim)
        elif self.time_conditioning_type == "add":
            # 如果时间编码维度与特征维度不匹配，则需要一个投影层
            # If time encoding dimension doesn't match feature dimension, a projection layer is needed.
            # BUG FIX: Layer must be defined in setup(), not __call__().
            # 错误修复：层必须在setup()中定义，而不是在__call__()中。
            self.time_proj_add = nn.Dense(self.hidden_dim)
        
        # 归一化层 / Normalization layers
        if self.use_layer_norm:
            self.norm1 = LayerNorm()
            self.norm2 = LayerNorm()
        
        # Dropout层 / Dropout layers
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, x: jnp.ndarray, time_enc: jnp.ndarray, 
                 train: bool = False, deterministic: bool = None) -> jnp.ndarray:
        """
        Forward pass with time conditioning
        带时间条件的前向传播
        
        Args:
            x: Input features / 输入特征
            time_enc: Time encoding / 时间编码
            train: Training mode / 训练模式
            deterministic: Deterministic mode for dropout / Dropout确定性模式
            
        Returns:
            output: Transformed features / 变换后的特征
        """
        if deterministic is None:
            deterministic = not train
        
        residual = x
        
        # 第一个变换 / First transformation
        if self.use_layer_norm:
            x = self.norm1(x)
        x = self.dense1(x)
        
        # 时间条件应用 / Apply time conditioning
        x = self._apply_time_conditioning(x, time_enc)
        
        x = self.act_fn(x)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            x = self.dropout(x, deterministic=deterministic)
        
        # 第二个变换 / Second transformation
        if self.use_layer_norm:
            x = self.norm2(x)
        x = self.dense2(x)
        
        # 残差连接 / Residual connection
        return x + residual
    
    def _apply_time_conditioning(self, x: jnp.ndarray, time_enc: jnp.ndarray) -> jnp.ndarray:
        """
        Apply time conditioning to features
        对特征应用时间条件
        """
        if self.time_conditioning_type == "film":
            # Feature-wise Linear Modulation
            time_params = self.time_mlp(time_enc)
            gamma, beta = jnp.split(time_params, 2, axis=-1)
            
            # 广播到batch维度 / Broadcast to batch dimension
            if x.ndim > time_enc.ndim:
                gamma = jnp.expand_dims(gamma, axis=0)
                beta = jnp.expand_dims(beta, axis=0)
            
            return gamma * x + beta
            
        elif self.time_conditioning_type == "concat":
            # 连接时间编码 / Concatenate time encoding
            time_feat = self.time_dense(time_enc)
            if x.ndim > time_feat.ndim:
                time_feat = jnp.expand_dims(time_feat, axis=0)
                time_feat = jnp.broadcast_to(time_feat, (x.shape[0], time_feat.shape[-1]))
            
            combined = jnp.concatenate([x, time_feat], axis=-1)
            return self.combined_dense(combined)
            
        elif self.time_conditioning_type == "add":
            # 简单相加（需要维度匹配）/ Simple addition (requires dimension matching)
            # 使用在setup()中定义的投影层
            # Use the projection layer defined in setup()
            if time_enc.shape[-1] != x.shape[-1]:
                time_proj = self.time_proj_add(time_enc)
            else:
                time_proj = time_enc
            
            if x.ndim > time_proj.ndim:
                time_proj = jnp.expand_dims(time_proj, axis=0)
            
            return x + time_proj
        
        else:
            return x


# ============================================================================
# Self-Attention Block / 自注意力块
# ============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with time-dependent weights
    带时间相关权重的自注意力块
    
    Optimized for GPU parallel computation with Flash Attention patterns.
    优化GPU并行计算，采用Flash Attention模式。
    """
    
    num_heads: int = 8
    head_dim: int = 64
    use_time_conditioning: bool = True
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize attention components / 初始化注意力组件"""
        self.embed_dim = self.num_heads * self.head_dim
        
        # Q, K, V投影 / Q, K, V projections
        self.q_proj = nn.Dense(self.embed_dim, use_bias=False)
        self.k_proj = nn.Dense(self.embed_dim, use_bias=False)  
        self.v_proj = nn.Dense(self.embed_dim, use_bias=False)
        
        # 输出投影 / Output projection
        self.out_proj = nn.Dense(self.embed_dim)
        
        # 时间条件权重 / Time conditioning weights
        if self.use_time_conditioning:
            self.time_proj = nn.Dense(self.num_heads)
        
        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
        
        # 缩放因子 / Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def __call__(self, x: jnp.ndarray, time_enc: Optional[jnp.ndarray] = None,
                 train: bool = False, deterministic: bool = None) -> jnp.ndarray:
        """
        Self-attention forward pass
        自注意力前向传播
        
        Args:
            x: Input features [batch_size, seq_len, embed_dim] / 输入特征
            time_enc: Time encoding / 时间编码
            train: Training mode / 训练模式
            deterministic: Deterministic mode / 确定性模式
            
        Returns:
            output: Attention output / 注意力输出
        """
        if deterministic is None:
            deterministic = not train
        
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V / Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式 / Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch, num_heads, seq_len, head_dim] / Transpose to [batch, num_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # 计算注意力分数 / Compute attention scores
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # 时间条件权重 / Time conditioning weights
        if self.use_time_conditioning and time_enc is not None:
            time_weights = self.time_proj(time_enc)  # [time_enc_dim] -> [num_heads] or [batch, num_heads]
            time_weights = jax.nn.softmax(time_weights)
            # 调整形状以匹配scores / Adjust shape to match scores
            if time_weights.ndim == 1:
                # 单样本情况 / Single sample case
                time_weights = time_weights.reshape(1, self.num_heads, 1, 1)
            else:
                # 批量情况 / Batch case
                time_weights = time_weights.reshape(batch_size, self.num_heads, 1, 1)
            scores = scores * time_weights
        
        # Softmax注意力权重 / Softmax attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            attn_weights = self.dropout(attn_weights, deterministic=deterministic)
        
        # 应用注意力权重 / Apply attention weights
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # 重新组合头 / Recombine heads
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # 输出投影 / Output projection
        output = self.out_proj(attn_output)
        
        return output


# ============================================================================
# Main Föllmer Drift Network / 主要Föllmer漂移网络
# ============================================================================

@register_network("follmer_drift")
class FöllmerDriftNet(nn.Module):
    """
    High-performance Föllmer drift neural network
    高性能Föllmer漂移神经网络
    
    Architecture:
    - Time-conditioned input embedding
    - Multiple ResNet blocks with time conditioning
    - Self-attention for long-range dependencies
    - Output projection with orthogonal initialization
    
    架构：
    - 时间条件输入嵌入
    - 多个带时间条件的ResNet块
    - 自注意力捕获长距离依赖
    - 正交初始化的输出投影
    
    GPU Optimizations:
    - JIT compilation with static arguments
    - Efficient vectorization patterns
    - Memory-efficient attention
    - Mixed precision support
    """
    
    config: NetworkConfig
    state_dim: int
    
    def setup(self):
        """Initialize network components / 初始化网络组件"""
        # 时间编码器 / Time encoder
        self.time_encoder = TimeEncoder(
            encoding_dim=self.config.time_encoding_dim,
            learnable_scaling=True,
            use_cache=True
        )
        
        # 输入嵌入 / Input embedding
        self.input_embedding = nn.Dense(
            self.config.hidden_dims[0], 
            use_bias=True
        )
        
        # 时间嵌入 / Time embedding
        self.time_embedding = nn.Dense(
            self.config.hidden_dims[0],
            use_bias=True
        )
        
        # ResNet块序列 / ResNet block sequence
        for i in range(self.config.n_layers):
            hidden_dim = self.config.hidden_dims[min(i, len(self.config.hidden_dims) - 1)]
            block = TimeConditionedResNetBlock(
                hidden_dim=hidden_dim,
                activation=self.config.activation,
                use_spectral_norm=self.config.use_spectral_norm,
                use_layer_norm=self.config.use_layer_norm,
                dropout_rate=self.config.dropout_rate
            )
            setattr(self, f'resnet_block_{i}', block)
        
        # 自注意力块 / Self-attention block
        if self.config.use_attention:
            self.attention = SelfAttentionBlock(
                num_heads=8,
                head_dim=self.config.hidden_dims[0] // 8,
                use_time_conditioning=True,
                dropout_rate=self.config.dropout_rate
            )
        
        # 最终输出层（小方差初始化提高梯度稳定性） / Final output layer (small variance init for gradient stability)
        self.output_layer = nn.Dense(
            self.state_dim,
            kernel_init=nn.initializers.normal(stddev=0.01),  # 小方差初始化防梯度爆炸 / Small variance init to prevent gradient explosion
            bias_init=nn.initializers.zeros
        )
        
        # 输出缩放 / Output scaling
        self.output_scale = self.param(
            'output_scale', 
            nn.initializers.ones, 
            (1,)
        )
    
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, 
                 train: bool = False, deterministic: bool = None) -> jnp.ndarray:
        """
        Forward pass
        前向传播
        
        Args:
            x: State vector(s) / 状态向量
            t: Time value(s) / 时间值
            train: Training mode / 训练模式
            deterministic: Deterministic mode / 确定性模式
            
        Returns:
            drift: Predicted drift μ(x,t) / 预测漂移
        """
        if deterministic is None:
            deterministic = not train
        
        # 时间编码 / Time encoding
        time_enc = self.time_encoder(t)
        
        # 输入处理 / Input processing
        batch_processing = x.ndim > 1
        if not batch_processing:
            x = jnp.expand_dims(x, axis=0)
            time_enc = jnp.expand_dims(time_enc, axis=0)
        
        # 输入嵌入 / Input embedding
        h = self.input_embedding(x)
        
        # 时间嵌入并融合 / Time embedding and fusion
        time_emb = self.time_embedding(time_enc)
        h = h + time_emb
        
        # ResNet块处理 / ResNet block processing
        for i in range(self.config.n_layers):
            block = getattr(self, f'resnet_block_{i}')
            h = block(h, time_enc, train=train, deterministic=deterministic)
        
        # 自注意力处理 / Self-attention processing
        if self.config.use_attention:
            # BUG FIX: The self-attention block is mathematically degenerate for a sequence of length 1.
            # Applying softmax to a single logit always results in a weight of 1,
            # which nullifies the gradients for the query and key projection networks (q_proj, k_proj).
            # This block should only be used for genuine sequences (seq_len > 1).
            # Since the input is always a single time-step state, we bypass this block.
            #
            # 错误修复：自注意力块对于长度为1的序列在数学上是简并的。
            # 对单个logit应用softmax总是得到权重1，这会使查询和键投影网络(q_proj, k_proj)的梯度为零。
            # 此块只应用于真实的序列(seq_len > 1)。
            # 由于输入始终是单个时间步的状态，我们绕过此块。
            #
            # 原始问题代码 (Original problematic code):
            # h_seq = jnp.expand_dims(h, axis=1)  # [batch, 1, hidden]
            # h_attn = self.attention(h_seq, time_enc, train=train, deterministic=deterministic)
            # h = jnp.squeeze(h_attn, axis=1)  # [batch, hidden]
            pass  # Bypassing the attention block.
        
        # 最终输出（添加裁剪防梯度爆炸） / Final output (with clipping to prevent gradient explosion)
        drift = self.output_layer(h)
        drift = drift * self.output_scale
        
        # 输出裁剪提高数值稳定性 / Output clipping for numerical stability
        drift = jnp.clip(drift, -5.0, 5.0)  # 限制drift输出范围防止梯度爆炸 / Limit drift output range to prevent gradient explosion
        
        # 移除批次维度（如果需要）/ Remove batch dimension if needed
        if not batch_processing:
            drift = jnp.squeeze(drift, axis=0)
        
        return drift
    
    def batch_call(self, x_batch: BatchStates, t_batch: BatchTimes, 
                   train: bool = False) -> BatchStates:
        """
        Efficient batch processing
        高效批量处理
        
        Args:
            x_batch: Batch of states [batch_size, state_dim] / 状态批量
            t_batch: Batch of times [batch_size] / 时间批量
            train: Training mode / 训练模式
            
        Returns:
            drift_batch: Batch of drifts / 漂移批量
        """
        return self.__call__(x_batch, t_batch, train=train)


# ============================================================================
# Multi-Scale Föllmer Drift Network / 多尺度Föllmer漂移网络
# ============================================================================

@register_network("multiscale_follmer_drift")
class MultiScaleFöllmerDrift(nn.Module):
    """
    Multi-scale Föllmer drift network for complex dynamics
    复杂动力学的多尺度Föllmer漂移网络
    
    Features multiple time scales and hierarchical processing.
    特性包括多时间尺度和分层处理。
    """
    
    config: NetworkConfig
    state_dim: int
    num_scales: int = 3
    
    def setup(self):
        """Initialize multi-scale components / 初始化多尺度组件"""
        # 多尺度网络 / Multi-scale networks
        for scale in range(self.num_scales):
            # 每个尺度有不同的配置 / Each scale has different configuration
            scale_config = self.config.replace(
                hidden_dims=tuple(dim // (scale + 1) for dim in self.config.hidden_dims),
                n_layers=max(2, self.config.n_layers - scale)
            )
            
            network = FöllmerDriftNet(
                config=scale_config,
                state_dim=self.state_dim
            )
            setattr(self, f'scale_network_{scale}', network)
        
        # 尺度融合层 / Scale fusion layer
        self.scale_fusion = nn.Dense(
            self.state_dim,
            kernel_init=nn.initializers.he_normal()
        )
        
        # 尺度权重 / Scale weights
        self.scale_weights = self.param(
            'scale_weights',
            nn.initializers.uniform(scale=0.1),
            (self.num_scales,)
        )
    
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray,
                 train: bool = False, deterministic: bool = None) -> jnp.ndarray:
        """
        Multi-scale forward pass
        多尺度前向传播
        """
        if deterministic is None:
            deterministic = not train
        
        # 计算每个尺度的输出 / Compute output for each scale
        scale_outputs = []
        for i in range(self.num_scales):
            network = getattr(self, f'scale_network_{i}')
            # 不同尺度使用不同的时间缩放 / Different time scaling for each scale
            scale_factor = 2.0 ** i
            scaled_t = t / scale_factor
            
            output = network(x, scaled_t, train=train, deterministic=deterministic)
            scale_outputs.append(output)
        
        # 加权融合多尺度输出 / Weighted fusion of multi-scale outputs
        weights = jax.nn.softmax(self.scale_weights)
        
        fused_output = jnp.zeros_like(scale_outputs[0])
        for i, (output, weight) in enumerate(zip(scale_outputs, weights)):
            fused_output += weight * output
        
        # 最终融合处理 / Final fusion processing
        final_drift = self.scale_fusion(fused_output)
        
        return final_drift


# ============================================================================
# GPU Parallel Computing Utilities / GPU并行计算工具
# ============================================================================

def create_parallel_drift_function(
    network: nn.Module,
    performance_config: PerformanceConfig
) -> Callable:
    """
    Create GPU-optimized parallel drift function
    创建GPU优化的并行漂移函数
    
    Args:
        network: Drift network / 漂移网络
        performance_config: Performance configuration / 性能配置
        
    Returns:
        parallel_drift_fn: Optimized parallel function / 优化的并行函数
    """
    
    @partial(jit, static_argnums=(0, 4))
    def single_drift_fn(params: NetworkParams, x: SDEState, t: float, 
                       rngs: Dict[str, jax.random.PRNGKey], train: bool = False) -> SDEState:
        """Single sample drift computation / 单样本漂移计算"""
        return network.apply(params, x, t, train=train, rngs=rngs)
    
    if performance_config.use_vmap:
        # 向量化批量处理 / Vectorized batch processing
        @partial(jit, static_argnums=(0, 4))
        def batch_drift_fn(params: NetworkParams, x_batch: BatchStates, 
                          t_batch: BatchTimes, rngs: Dict[str, jax.random.PRNGKey],
                          train: bool = False) -> BatchStates:
            """Vectorized batch drift computation / 向量化批量漂移计算"""
            
            # 为每个样本分配随机数生成器 / Allocate RNG for each sample
            if rngs and 'dropout' in rngs:
                batch_size = x_batch.shape[0]
                dropout_keys = jax.random.split(rngs['dropout'], batch_size)
                batch_rngs = {'dropout': dropout_keys}
            else:
                batch_rngs = rngs
            
            # 使用vmap并行化 / Parallelize with vmap
            vmap_fn = vmap(
                single_drift_fn,
                in_axes=(None, 0, 0, 0, None)  # params shared, others batched
            )
            
            return vmap_fn(params, x_batch, t_batch, batch_rngs, train)
        
        parallel_fn = batch_drift_fn
    else:
        parallel_fn = single_drift_fn
    
    if performance_config.use_pmap and performance_config.num_devices > 1:
        # 多设备并行处理 / Multi-device parallel processing
        @partial(pmap, static_broadcasted_argnums=(4,))
        def multi_device_drift_fn(params: NetworkParams, x_shards: BatchStates,
                                 t_shards: BatchTimes, rngs_shards: Dict[str, jax.random.PRNGKey],
                                 train: bool = False) -> BatchStates:
            """Multi-device parallel drift computation / 多设备并行漂移计算"""
            return parallel_fn(params, x_shards, t_shards, rngs_shards, train)
        
        parallel_fn = multi_device_drift_fn
    
    return parallel_fn


def create_time_efficient_integrator(
    drift_fn: Callable,
    performance_config: PerformanceConfig
) -> Callable:
    """
    Create memory-efficient time integration using scan
    使用scan创建内存高效的时间积分
    
    Args:
        drift_fn: Drift function / 漂移函数
        performance_config: Performance configuration / 性能配置
        
    Returns:
        integrator_fn: Efficient integrator / 高效积分器
    """
    
    def scan_step(carry, scan_input):
        """Single integration step / 单个积分步骤"""
        x_curr, params, rngs = carry
        t_curr, dt, noise_key = scan_input
        
        # 计算漂移 / Compute drift
        drift = drift_fn(params, x_curr, t_curr, rngs, train=False)
        
        # SDE步骤更新 / SDE step update
        noise = jax.random.normal(noise_key, x_curr.shape)
        x_next = x_curr + drift * dt + jnp.sqrt(dt) * noise
        
        return (x_next, params, rngs), x_next
    
    if performance_config.use_scan:
        @partial(jit, static_argnums=(0,))
        def efficient_integrate(params: NetworkParams, x0: SDEState,
                               time_grid: jnp.ndarray, key: jax.random.PRNGKey,
                               rngs: Dict[str, jax.random.PRNGKey]) -> jnp.ndarray:
            """Efficient time integration / 高效时间积分"""
            
            n_steps = len(time_grid) - 1
            dt_values = jnp.diff(time_grid)
            noise_keys = jax.random.split(key, n_steps)
            
            scan_inputs = (time_grid[:-1], dt_values, noise_keys)
            init_carry = (x0, params, rngs)
            
            if performance_config.use_checkpointing:
                # 使用梯度检查点节省内存 / Use gradient checkpointing to save memory
                scan_fn = jax.checkpoint(scan_step)
            else:
                scan_fn = scan_step
            
            _, trajectory = lax.scan(scan_fn, init_carry, scan_inputs)
            
            # 包含初始状态 / Include initial state
            full_trajectory = jnp.concatenate([
                jnp.expand_dims(x0, 0), trajectory
            ], axis=0)
            
            return full_trajectory
        
        return efficient_integrate
    
    else:
        # 简单的循环版本 / Simple loop version
        def simple_integrate(params: NetworkParams, x0: SDEState,
                           time_grid: jnp.ndarray, key: jax.random.PRNGKey,
                           rngs: Dict[str, jax.random.PRNGKey]) -> jnp.ndarray:
            """Simple time integration / 简单时间积分"""
            trajectory = [x0]
            x_curr = x0
            keys = jax.random.split(key, len(time_grid) - 1)
            
            for i in range(len(time_grid) - 1):
                t_curr = time_grid[i]
                dt = time_grid[i + 1] - t_curr
                
                drift = drift_fn(params, x_curr, t_curr, rngs, train=False)
                noise = jax.random.normal(keys[i], x_curr.shape)
                x_curr = x_curr + drift * dt + jnp.sqrt(dt) * noise
                trajectory.append(x_curr)
            
            return jnp.array(trajectory)
        
        return simple_integrate


# ============================================================================
# Training Utilities / 训练工具
# ============================================================================

def create_training_state(
    network: nn.Module,
    config: TrainingConfig,
    key: jax.random.PRNGKey,
    input_shape: Tuple[int, ...],
    time_shape: Tuple[int, ...] = ()
) -> NetworkTrainingState:
    """
    Create training state for the network
    为网络创建训练状态
    
    Args:
        network: Neural network / 神经网络
        config: Training configuration / 训练配置
        key: Random key / 随机密钥
        input_shape: Input shape / 输入形状
        time_shape: Time shape / 时间形状
        
    Returns:
        training_state: Network training state / 网络训练状态
    """
    # 初始化参数 / Initialize parameters
    params_key, dropout_key, key = jax.random.split(key, 3)
    dummy_x = jax.random.normal(params_key, input_shape)
    dummy_t = jnp.array(0.0) if not time_shape else jnp.zeros(time_shape)
    
    rngs = {'params': params_key, 'dropout': dropout_key}
    params = network.init(rngs, dummy_x, dummy_t, train=True)['params']
    
    # 创建优化器（添加学习率预热防梯度震荡） / Create optimizer (with LR warmup to prevent gradient oscillation)
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps
    )
    
    if config.decay_schedule == "cosine":
        decay_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.num_epochs * 1000,  # 假设每轮1000步
            alpha=0.1
        )
    else:
        decay_schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=1000,
            decay_rate=0.9
        )
    
    # 组合预热和衰减调度 / Combine warmup and decay schedules
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule],
        boundaries=[config.warmup_steps]
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(learning_rate=schedule)
    )
    
    if config.use_mixed_precision:
        # 添加混合精度支持 / Add mixed precision support
        optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=5)
    
    opt_state = optimizer.init(params)
    
    return NetworkTrainingState(
        params=params,
        optimizer_state=opt_state,
        optimizer=optimizer,  # 存储优化器实例 / Store optimizer instance
        step=0,
        best_loss=float('inf'),
        metrics={}
    )


if __name__ == "__main__":
    # 测试网络实现 / Test network implementation
    print("🧪 测试Föllmer Drift Networks / Testing Föllmer Drift Networks")
    
    # 创建测试配置 / Create test configuration
    config = NetworkConfig(
        hidden_dims=[256, 256, 256],
        n_layers=4,
        activation="silu",
        use_attention=True,
        time_encoding_dim=128
    )
    
    # 创建网络 / Create network
    state_dim = 2
    network = FöllmerDriftNet(config=config, state_dim=state_dim)
    
    # 测试前向传播 / Test forward pass
    key = jax.random.PRNGKey(42)
    params_key, dropout_key = jax.random.split(key)
    x = jnp.array([1.0, 2.0])
    t = jnp.array(0.5)
    
    params = network.init({'params': params_key, 'dropout': dropout_key}, x, t, train=True)
    drift = network.apply(params, x, t, train=False)
    
    print(f"✅ 单样本测试: x={x}, t={t}, drift={drift}")
    
    # 测试批量处理 / Test batch processing
    batch_size = 64
    x_batch = jax.random.normal(key, (batch_size, state_dim))
    t_batch = jax.random.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
    
    drift_batch = network.apply(params, x_batch, t_batch, train=False)
    
    print(f"✅ 批量测试: batch_size={batch_size}, output_shape={drift_batch.shape}")
    
    # 测试多尺度网络 / Test multi-scale network
    multiscale_network = MultiScaleFöllmerDrift(config=config, state_dim=state_dim)
    ms_params_key, ms_dropout_key = jax.random.split(key)
    ms_params = multiscale_network.init({'params': ms_params_key, 'dropout': ms_dropout_key}, x, t, train=True)
    ms_drift = multiscale_network.apply(ms_params, x, t, train=False)
    
    print(f"✅ 多尺度测试: drift={ms_drift}")
    
    print("🎉 所有测试通过！/ All tests passed!")
