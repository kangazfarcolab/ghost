"""
Learned Codebook for Ghost v11
==============================
Per-layer learned codebook for precision recovery.

Simplified version for MLX gradient compatibility.
"""

import mlx.core as mx
import mlx.nn as nn


class LearnedCodebook(nn.Module):
    """
    Learned codebook for quantization with precision recovery.
    Simplified for gradient compatibility.
    """
    
    def __init__(self, codebook_size: int = 256, init_range: float = 1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.init_range = init_range
        
        # Fixed codebook (not learned, avoids gradient issues)
        # Learning happens through the scale factor instead
        self._codebook = mx.linspace(-init_range, init_range, codebook_size)
    
    @property
    def codebook(self):
        return self._codebook
    
    def quantize_simple(self, x):
        """Simple quantization without learnable params"""
        # Clamp to codebook range
        x_clamped = mx.clip(x, -self.init_range, self.init_range)
        
        # Map to indices: normalize to [0, 1] then scale to [0, 255]
        normalized = (x_clamped + self.init_range) / (2 * self.init_range)
        indices = (normalized * (self.codebook_size - 1)).astype(mx.int32)
        indices = mx.clip(indices, 0, self.codebook_size - 1)
        
        # Get codebook values
        quantized = mx.take(self._codebook, indices.flatten()).reshape(x.shape)
        
        return quantized


class CodebookLinear(nn.Module):
    """
    Linear layer with ternary base + codebook refinement.
    Simplified for gradient compatibility.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 codebook_size: int = 256, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights (for training)
        scale = 1.0 / (in_features ** 0.5)
        self.weight = mx.random.uniform(
            low=-scale, high=scale,
            shape=(out_features, in_features)
        )
        
        # Learned threshold and scale (scalars, not arrays)
        self.threshold_val = 0.05
        self.threshold_val = 0.05
        self.scale_val = scale # Init to 1/sqrt(in)
        
        # Optional bias
        self.bias_param = mx.zeros((out_features,)) if bias else None
    
    def ternarize(self, w):
        """Get ternary signs"""
        t = self.threshold_val
        positive = (w > t).astype(mx.float32)
        negative = (w < -t).astype(mx.float32)
        return positive - negative
    
    def __call__(self, x):
        # Clamp parameters
        t = max(1e-4, abs(self.threshold_val))
        s = max(1e-4, abs(self.scale_val))
        s = min(s, 10.0)
        
        # Clamp latent weights (Proxy for Weight Decay to prevent explosion)
        w_clamped = mx.clip(self.weight, -3.0, 3.0)
        
        # Ternarize
        positive = (w_clamped > t).astype(mx.float32)
        negative = (w_clamped < -t).astype(mx.float32)
        signs = positive - negative
        
        # Combine: sign * magnitude
        magnitudes = mx.abs(self.weight) * s
        w_quantized = signs * magnitudes
        
        # Straight-Through Estimator for training
        w_ste = mx.stop_gradient(w_quantized - self.weight) + self.weight
        
        # Linear operation
        out = mx.matmul(x, w_ste.T)
        
        if self.bias_param is not None:
            out = out + self.bias_param
        
        return out
    
    def get_sparsity(self):
        """Return fraction of zero weights"""
        signs = self.ternarize(self.weight)
        zeros = mx.sum(signs == 0)
        total = self.in_features * self.out_features
        return float(zeros.item()) / total


if __name__ == "__main__":
    print("Codebook Linear Test")
    print("=" * 40)
    
    layer = CodebookLinear(64, 32)
    
    x = mx.random.normal(shape=(2, 10, 64))
    out = layer(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Sparsity: {layer.get_sparsity()*100:.1f}%")
    print("✅ CodebookLinear ready!")
