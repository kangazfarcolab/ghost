"""
Ternary Mamba for Ghost v12
===========================
State Space Model with ternary weights + codebook.
Simplified SSM for MLX gradient compatibility.
"""

import mlx.core as mx
import mlx.nn as nn
import math

from ghost_model_v12.core.ternary_linear import TernaryLinear
from ghost_model_v12.core.learned_codebook import CodebookLinear


class TernaryMambaConfig:
    """Configuration for TernaryMamba"""
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2


class TernaryMamba(nn.Module):
    """
    State Space Model with ternary weights.
    Simplified for MLX gradient compatibility.
    Uses a single-pass approximation instead of sequential loop.
    """
    
    def __init__(self, dim: int, config: TernaryMambaConfig = None):
        super().__init__()
        config = config or TernaryMambaConfig()
        
        self.dim = dim
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = dim * config.expand
        
        # Ternary projections
        self.in_proj = TernaryLinear(dim, self.d_inner * 2)
        self.out_proj = TernaryLinear(self.d_inner, dim)
        
        # Conv1D for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner
        )
        
        # Simplified SSM: use attention-like mechanism instead of recurrence
        # This is more gradient-friendly
        self.dt_proj = TernaryLinear(self.d_inner, self.d_inner)
        self.A = TernaryLinear(self.d_inner, self.d_inner)
        self.D = mx.ones((self.d_inner,))
    
    def ssm_simple(self, x):
        """
        Simplified SSM using attention-like decay.
        More gradient-friendly than sequential loop.
        """
        B, L, D = x.shape
        
        # Compute dynamics
        dt = mx.sigmoid(self.dt_proj(x))  # [B, L, D] decay rates
        
        # Create causal decay mask
        positions = mx.arange(L)[None, :, None]  # [1, L, 1]
        positions_t = mx.arange(L)[None, None, :]  # [1, 1, L]
        
        # Decay based on distance (causal: only look back)
        distances = positions - positions_t  # [1, L, L]
        causal_mask = (distances >= 0).astype(mx.float32)
        decay = mx.exp(-0.1 * distances.astype(mx.float32)) * causal_mask  # [1, L, L]
        
        # Apply A transformation
        Ax = self.A(x)  # [B, L, D]
        
        # Weighted sum with decay (like attention)
        # y[t] = sum_{s<=t} decay[t,s] * Ax[s]
        y = mx.matmul(decay, Ax)  # [B, L, D]
        
        # Add skip connection
        y = y + x * self.D[None, None, :]
        
        return y
    
    def __call__(self, x):
        B, L, D = x.shape
        
        # Project and split into x and gate
        xz = self.in_proj(x)
        x_branch, z = mx.split(xz, 2, axis=-1)
        
        # Conv1D
        x_conv = self.conv1d(x_branch)[:, :L, :]
        x_conv = nn.silu(x_conv)
        
        # SSM (simplified)
        y = self.ssm_simple(x_conv)
        
        # Gate and output
        y = y * nn.silu(z)
        out = self.out_proj(y)
        
        return out
    
    def step(self, x, cache=None):
        """
        Step-by-step inference with state (KV-cache equivalent).
        x: [B, 1, D]
        cache: (conv_state, ssm_state)
        """
        B, _, D = x.shape
        if cache is None:
            # Init empty cache: conv buffer, ssm state
            conv_state = mx.zeros((B, self.d_conv - 1, self.d_inner))
            ssm_state = mx.zeros((B, 1, self.d_inner))
            cache = (conv_state, ssm_state)
            
        conv_state, ssm_state = cache
        
        # 1. Projection
        xz = self.in_proj(x)
        x_branch, z = mx.split(xz, 2, axis=-1)
        
        # 2. Conv Step
        # Concatenate buffer + current -> [B, d_conv, d_inner]
        conv_input = mx.concatenate([conv_state, x_branch], axis=1)
        
        # Apply Conv1D (causal) and take last step
        # Input length K, output length K (due to padding=K-1 and slicing usually)
        # But slicing happens in __call__, here we rely on MLX conv1d behavior or slicing ourselves
        conv_out_full = self.conv1d(conv_input)
        
        # We want the output corresponding to the full window (the last element)
        # With padding=K-1, the first output sees 1 element, the K-th output sees K elements.
        # Examples: K=4. Input indices 0,1,2,3.
        # Out[0] sees [0,0,0,0] (padded) + [0]
        # Out[3] sees [0,1,2,3] -> accurate.
        conv_out = conv_out_full[:, -1:, :]
        x_conv = nn.silu(conv_out)
        
        # Update buffer (shift left)
        new_conv_state = conv_input[:, 1:, :]
        
        # 3. SSM Step (Recurrent)
        # y_t = (Ax)_t + decay * y_{t-1}
        decay_factor = math.exp(-0.1)
        
        Ax = self.A(x_conv)
        new_ssm_state = Ax + decay_factor * ssm_state
        
        y = new_ssm_state + x_conv * self.D[None, None, :]
        
        # 4. Gate + Out
        y = y * nn.silu(z)
        out = self.out_proj(y)
        
        return out, (new_conv_state, new_ssm_state)


if __name__ == "__main__":
    print("TernaryMamba Test")
    print("=" * 40)
    
    mamba = TernaryMamba(dim=64)
    mx.eval(mamba.parameters())
    
    x = mx.random.normal(shape=(2, 16, 64))
    out = mamba(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    # Test gradient
    def loss_fn(m):
        return mx.mean(m(x))
    
    loss, grads = mx.value_and_grad(loss_fn)(mamba)
    print(f"Gradient loss: {loss.item():.4f}")
    print("✅ TernaryMamba ready!")
