"""
Ghost v20 - Multi-Digit Math (Learned Carry)
=============================================
Extend v20 to handle ANY size numbers through:
1. Digit parsing (break "1000" into [1,0,0,0])
2. Learned carry table (a+b+carry → result, new_carry)
3. Reverse processing (right to left)

All operations are LEARNED in weights, not hardcoded!
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time


class LearnedCarryTable(nn.Module):
    """
    Learn the full addition with carry.
    
    Input: digit_a (0-9), digit_b (0-9), carry_in (0-1)
    Output: result (0-9), carry_out (0-1)
    
    Examples:
    - 5 + 3 + 0 = 8, carry 0
    - 5 + 7 + 0 = 2, carry 1  (12 → 2 with carry)
    - 9 + 9 + 1 = 9, carry 1  (19 → 9 with carry)
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.digit_embed = nn.Embedding(10, hidden_dim)  # 0-9
        self.carry_embed = nn.Embedding(2, hidden_dim)   # 0 or 1
        
        # Network: 3 embeddings → result (0-9) + carry (0-1)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output heads
        self.result_head = nn.Linear(hidden_dim, 10)  # 0-9
        self.carry_head = nn.Linear(hidden_dim, 2)    # 0 or 1
    
    def forward_batch(self, a, b, carry):
        """Batch forward for training"""
        emb_a = self.digit_embed(a)
        emb_b = self.digit_embed(b)
        emb_c = self.carry_embed(carry)
        
        combined = mx.concatenate([emb_a, emb_b, emb_c], axis=-1)
        h = nn.relu(self.fc1(combined))
        h = nn.relu(self.fc2(h))
        
        result_logits = self.result_head(h)
        carry_logits = self.carry_head(h)
        
        return result_logits, carry_logits
    
    def add_with_carry(self, a: int, b: int, carry_in: int) -> tuple:
        """Single addition with carry (inference)"""
        a_arr = mx.array([a], dtype=mx.int32)
        b_arr = mx.array([b], dtype=mx.int32)
        c_arr = mx.array([carry_in], dtype=mx.int32)
        
        result_logits, carry_logits = self.forward_batch(a_arr, b_arr, c_arr)
        
        result = int(mx.argmax(result_logits[0]).item())
        carry_out = int(mx.argmax(carry_logits[0]).item())
        
        return result, carry_out
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


class MultiDigitMath(nn.Module):
    """
    Complete multi-digit arithmetic using learned tables.
    
    Supports:
    - Addition of any size numbers
    - Subtraction (todo)
    - Multiplication (todo)
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.carry_table = LearnedCarryTable(hidden_dim)
    
    def parse_number(self, s: str) -> list:
        """Parse number string to digit list (reversed for processing)"""
        return [int(c) for c in s.strip()][::-1]  # Reverse for right-to-left
    
    def digits_to_string(self, digits: list) -> str:
        """Convert digit list back to string"""
        # Remove leading zeros and reverse back
        result = digits[::-1]
        while len(result) > 1 and result[0] == 0:
            result = result[1:]
        return "".join(str(d) for d in result)
    
    def add(self, num1: str, num2: str) -> str:
        """Add two multi-digit numbers"""
        # Parse to reversed digit lists
        d1 = self.parse_number(num1)
        d2 = self.parse_number(num2)
        
        # Pad to same length
        max_len = max(len(d1), len(d2))
        d1 = d1 + [0] * (max_len - len(d1))
        d2 = d2 + [0] * (max_len - len(d2))
        
        # Add digit by digit with carry
        result = []
        carry = 0
        
        for i in range(max_len):
            digit_result, carry = self.carry_table.add_with_carry(d1[i], d2[i], carry)
            result.append(digit_result)
        
        # Handle final carry
        if carry > 0:
            result.append(carry)
        
        return self.digits_to_string(result)
    
    def count_params(self):
        return self.carry_table.count_params()


def create_carry_training_data():
    """Create all possible carry combinations"""
    data = []
    for a in range(10):
        for b in range(10):
            for carry_in in range(2):
                total = a + b + carry_in
                result = total % 10
                carry_out = 1 if total >= 10 else 0
                data.append((a, b, carry_in, result, carry_out))
    return data


def train_carry_table(model, steps=3000, lr=1e-2):
    """Train the carry table"""
    print("📐 Training Carry Table...")
    print("-" * 50)
    
    optimizer = optim.AdamW(learning_rate=lr)
    DATA = create_carry_training_data()
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        carry_in = mx.array([x[2] for x in batch], dtype=mx.int32)
        result_target = mx.array([x[3] for x in batch], dtype=mx.int32)
        carry_target = mx.array([x[4] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.carry_table.update(params)
            result_logits, carry_logits = model.carry_table.forward_batch(a, b, carry_in)
            
            # Cross entropy for result
            result_log_probs = mx.log(mx.softmax(result_logits, axis=-1) + 1e-10)
            result_loss = -mx.mean(mx.take_along_axis(result_log_probs, result_target[:, None], axis=1))
            
            # Cross entropy for carry
            carry_log_probs = mx.log(mx.softmax(carry_logits, axis=-1) + 1e-10)
            carry_loss = -mx.mean(mx.take_along_axis(carry_log_probs, carry_target[:, None], axis=1))
            
            return result_loss + carry_loss
        
        loss, grads = mx.value_and_grad(loss_fn)(model.carry_table.trainable_parameters())
        optimizer.update(model.carry_table, grads)
        mx.eval(model.carry_table.parameters())
        
        if step % 1000 == 0 or step == steps - 1:
            print(f"   Step {step:4d} | Loss: {float(loss):.6f}")
    
    # Test accuracy
    correct = 0
    for a, b, c_in, exp_result, exp_carry in DATA:
        result, carry = model.carry_table.add_with_carry(a, b, c_in)
        if result == exp_result and carry == exp_carry:
            correct += 1
    
    print(f"   Carry table accuracy: {correct}/{len(DATA)} ({correct/len(DATA)*100:.1f}%)")
    return correct == len(DATA)


def test_multi_digit(model):
    """Test multi-digit addition"""
    print("\n🧮 Testing Multi-Digit Addition:")
    print("-" * 50)
    
    tests = [
        ("3", "5", "8"),              # Single digit
        ("9", "9", "18"),             # Carry
        ("12", "34", "46"),           # Two digit
        ("99", "1", "100"),           # Cascade carry
        ("123", "456", "579"),        # Three digit
        ("999", "1", "1000"),         # Max cascade
        ("1000", "10000", "11000"),   # The target case!
        ("12345", "67890", "80235"),  # Five digit
        ("999999", "1", "1000000"),   # Six digit cascade
    ]
    
    correct = 0
    for a, b, expected in tests:
        result = model.add(a, b)
        status = "✅" if result == expected else "❌"
        print(f"   {a} + {b} = {result} (expected: {expected}) {status}")
        if result == expected:
            correct += 1
    
    print(f"\n   Accuracy: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")
    return correct / len(tests)


def main():
    print("=" * 60)
    print("     GHOST v20 - MULTI-DIGIT MATH")
    print("=" * 60)
    
    # Create model
    model = MultiDigitMath(hidden_dim=32)
    mx.eval(model.carry_table.parameters())
    
    print(f"\n📦 Carry Table: {model.count_params():,} params")
    print(f"   Training data: 200 combinations (10×10×2)")
    
    # Train
    start = time.perf_counter()
    success = train_carry_table(model, steps=3000)
    train_time = time.perf_counter() - start
    
    print(f"\n⏱️ Training time: {train_time:.1f}s")
    
    # Test
    accuracy = test_multi_digit(model)
    
    # Speed test
    print("\n⚡ Speed Test:")
    start = time.perf_counter()
    for _ in range(1000):
        model.add("12345", "67890")
    elapsed = time.perf_counter() - start
    print(f"   5-digit addition: {1000/elapsed:.0f} ops/s")
    
    start = time.perf_counter()
    for _ in range(1000):
        model.add("1000000", "9999999")
    elapsed = time.perf_counter() - start
    print(f"   7-digit addition: {1000/elapsed:.0f} ops/s")
    
    print("\n" + "=" * 60)
    print("                 SUMMARY")
    print("=" * 60)
    print(f"   Params: {model.count_params():,}")
    print(f"   Training: {train_time:.1f}s")
    print(f"   Accuracy: {accuracy*100:.0f}%")
    
    if accuracy == 1.0:
        print("\n   ✅ MULTI-DIGIT MATH WORKING!")
        print("   Now handles 1000+10000=11000 and beyond!")
    else:
        print("\n   ⚠️ Some errors - need debugging")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
