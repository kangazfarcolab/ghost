import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map

def zeroth_power_via_newton_schulz5(G, steps=5, eps=1e-7):
    """
    Compute the zeroth power of a matrix (orthogonalization) using Newton-Schulz iteration.
    X_0 = G
    X_{k+1} = 1.5 * X_k - 0.5 * X_k @ (X_k.T @ X_k)  (approx inverse square root normalization)
    
    This is the core of Muon: passing gradients through this filter forces them to be orthogonal.
    """
    # G shape: [Rows, Cols] or [Rows, Cols, ...]. Flatten to 2D
    original_shape = G.shape
    if len(original_shape) > 2:
        # Flatten: [Out, In] e.g. for Conv [Out, In, K] -> [Out, In*K]
        G = G.reshape(original_shape[0], -1)
    
    rows, cols = G.shape
    
    # Pre-normalize to ensure spectral norm < sqrt(5/3) for convergence
    trace = mx.sum(G * G).item()
    norm = (trace) ** 0.5
    X = G / (norm + eps)
    
    # Newton-Schulz Iteration (Order 2 approx is quicker than Order 5 usually)
    # Using the variant: X = X * (3I - X^T X) / 2
    
    for _ in range(steps):
        A = X.T @ X
        B = 3.0 * mx.eye(cols, dtype=X.dtype) - A
        X = 0.5 * X @ B
        
    # Reshape back
    if len(original_shape) > 2:
        X = X.reshape(original_shape)
        
    return X

class Muon(optim.Optimizer):
    """
    Muon: Momentum Orthogonal Optimizer.
    Designed for neural networks to force feature diversity.
    
    update = -lr * orthogonalize(momentum) * scaling_factor
    """
    def __init__(self, learning_rate, momentum=0.95, weight_decay=0.0, nesterov=True, ns_steps=5):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        # self.state is already initialized by super().__init__()

    def apply_gradients(self, gradients, model):
        from mlx.utils import tree_flatten, tree_unflatten
        
        # Flatten structure
        flat_params = tree_flatten(model.parameters())
        flat_grads = tree_flatten(gradients)
        
        # We must collect values in the exact order of flat_params to unflatten later
        updates = []
        
        for (name, param), (g_name, grad) in zip(flat_params, flat_grads):
            # 1. State Init
            if name not in self.state:
                self.state[name] = mx.zeros_like(param)
            
            buf = self.state[name]
            
            # 2. Weight Decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
                
            # 3. Momentum
            buf = self.momentum * buf + grad
            self.state[name] = buf
            
            # 4. Update
            if len(param.shape) >= 2:
                # Muon Orthogonal Update
                update_src = (grad + self.momentum * buf) if self.nesterov else buf
                ortho_update = zeroth_power_via_newton_schulz5(update_src, steps=self.ns_steps)
                new_param = param - self.learning_rate * ortho_update
            else:
                # SGD Update
                update_src = (grad + self.momentum * buf) if self.nesterov else buf
                new_param = param - self.learning_rate * update_src
                
            updates.append((name, new_param))
            
        # Unflatten
        return tree_unflatten(updates)
