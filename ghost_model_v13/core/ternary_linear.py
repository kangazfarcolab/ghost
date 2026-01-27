"""
Ternary Linear Layer for Ghost v11
==================================
Weights are quantized to {-1, 0, +1} with learned thresholds.

Benefits:
- 2 bits per weight (vs 1-bit binary)
- Sparsity from zeros
- Better gradient flow than binary
"""

import mlx.core as mx
import mlx.nn as nn


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights (-1, 0, +1).
    
    During training: uses full precision with STE (Straight-Through Estimator)
    During inference: uses ternary weights with scale factor
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights (for training)
        scale = 1.0 / (in_features ** 0.5)
        self.weight = mx.random.uniform(
            low=-scale, high=scale,
            shape=(out_features, in_features)
        )
        
        # Learned threshold for ternarization (per output channel)
        self.threshold = mx.ones((out_features, 1)) * 0.05
        
        # Learned scale factor (per output channel)
        # Init to 1/sqrt(in) to preserve variance, not 1.0!
        init_scale = 1.0 / (in_features ** 0.5)
        self.scale = mx.full((out_features, 1), init_scale)
        
        # Optional bias
        self.bias = mx.zeros((out_features,)) if bias else None
    
    def ternarize(self, w):
        """Convert weights to ternary {-1, 0, +1}"""
        t = mx.abs(self.threshold)
        
        positive = (w > t).astype(mx.float32)
        negative = (w < -t).astype(mx.float32)
        
        # +1 if positive, -1 if negative, 0 otherwise
        return positive - negative
    
    def __call__(self, x):
        # Clamp weights to prevent explosion (matches CodebookLinear)
        w_clamped = mx.clip(self.weight, -3.0, 3.0)
        
        # Get ternary weights
        # Clamp threshold to avoid degeneration
        t = mx.clip(mx.abs(self.threshold), 1e-4, 1.0)
        
        positive = (w_clamped > t).astype(mx.float32)
        negative = (w_clamped < -t).astype(mx.float32)
        w_ternary = positive - negative
        
        # Apply learned scale (clamped)
        # Prevent scale from exploding or vanishing
        s = mx.clip(mx.abs(self.scale), 1e-4, 10.0)
        w_scaled = w_ternary * s
        
        # Straight-Through Estimator: use ternary forward, gradient through clamped
        w_ste = mx.stop_gradient(w_scaled - w_clamped) + w_clamped
        
        # Linear operation
        out = mx.matmul(x, w_ste.T)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def get_sparsity(self):
        """Return fraction of zero weights"""
        w_ternary = self.ternarize(self.weight)
        zeros = mx.sum(w_ternary == 0)
        total = self.in_features * self.out_features
        return float(zeros.item()) / total
    
    def count_bits(self):
        """Count effective bits for storage"""
        # Ternary = 2 bits per weight (approximate, could be 1.58 with entropy coding)
        return self.in_features * self.out_features * 2


class TernaryWeight:
    """
    Helper class for storing/loading ternary weights.
    Packs 4 ternary values into 1 byte.
    """
    
    @staticmethod
    def pack(weights):
        """Pack ternary weights (-1, 0, 1) into bytes"""
        # Map: -1 -> 0, 0 -> 1, 1 -> 2
        mapped = (weights + 1).astype(mx.int32)
        
        # Pack 4 values per byte (2 bits each)
        flat = mapped.flatten()
        padded_len = ((len(flat) + 3) // 4) * 4
        padded = mx.pad(flat, [(0, padded_len - len(flat))])
        
        # Combine 4 values: v0 + v1*4 + v2*16 + v3*64
        reshaped = padded.reshape(-1, 4)
        packed = reshaped[:, 0] + reshaped[:, 1] * 4 + reshaped[:, 2] * 16 + reshaped[:, 3] * 64
        
        return packed.astype(mx.uint8)
    
    @staticmethod
    def unpack(packed, shape):
        """Unpack bytes back to ternary weights"""
        # Extract 4 values per byte
        v0 = packed % 4
        v1 = (packed // 4) % 4
        v2 = (packed // 16) % 4
        v3 = (packed // 64) % 4
        
        # Interleave and reshape
        unpacked = mx.stack([v0, v1, v2, v3], axis=1).flatten()
        unpacked = unpacked[:shape[0] * shape[1]].reshape(shape)
        
        # Unmap: 0 -> -1, 1 -> 0, 2 -> 1
        return (unpacked - 1).astype(mx.float32)


if __name__ == "__main__":
    print("TernaryLinear Test")
    print("=" * 40)
    
    layer = TernaryLinear(64, 32)
    
    x = mx.random.normal(shape=(2, 10, 64))
    out = layer(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Sparsity: {layer.get_sparsity()*100:.1f}%")
    print(f"Bits: {layer.count_bits():,} ({layer.count_bits()/8/1024:.2f} KB)")
    
    # Test packing
    w_ternary = layer.ternarize(layer.weight)
    packed = TernaryWeight.pack(w_ternary)
    unpacked = TernaryWeight.unpack(packed, w_ternary.shape)
    
    match = mx.all(w_ternary == unpacked)
    print(f"Pack/Unpack match: {bool(match.item())}")
    print(f"Packed size: {len(packed)} bytes vs {w_ternary.size * 4} bytes (float32)")
    print("✅ TernaryLinear ready!")
