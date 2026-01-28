"""
Ghost v20 Improved - With Generated Data & Retry Training
==========================================================
Enhanced training with:
1. Auto-generated datasets (3000+ examples)
2. Retry training until 100% on math tables
3. Better LoRA training with larger datasets
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time
import re
import os
import sys
from typing import List, Tuple, Dict, Optional

# Import generators
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.generators import generate_code_data, generate_logic_data, generate_fact_data


# ============================================================
# MATH TABLES (same as ghost_v20.py)
# ============================================================

class LearnedCarryTable(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.digit_embed = nn.Embedding(10, hidden_dim)
        self.carry_embed = nn.Embedding(2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.result_head = nn.Linear(hidden_dim, 10)
        self.carry_head = nn.Linear(hidden_dim, 2)
    
    def forward_batch(self, a, b, carry):
        emb = mx.concatenate([self.digit_embed(a), self.digit_embed(b), self.carry_embed(carry)], axis=-1)
        h = nn.relu(self.fc2(nn.relu(self.fc1(emb))))
        return self.result_head(h), self.carry_head(h)
    
    def add_with_carry(self, a: int, b: int, carry_in: int) -> tuple:
        r, c = self.forward_batch(mx.array([a]), mx.array([b]), mx.array([carry_in]))
        return int(mx.argmax(r[0]).item()), int(mx.argmax(c[0]).item())


class LearnedBorrowTable(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.digit_embed = nn.Embedding(10, hidden_dim)
        self.borrow_embed = nn.Embedding(2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.result_head = nn.Linear(hidden_dim, 10)
        self.borrow_head = nn.Linear(hidden_dim, 2)
    
    def forward_batch(self, a, b, borrow):
        emb = mx.concatenate([self.digit_embed(a), self.digit_embed(b), self.borrow_embed(borrow)], axis=-1)
        h = nn.relu(self.fc2(nn.relu(self.fc1(emb))))
        return self.result_head(h), self.borrow_head(h)
    
    def sub_with_borrow(self, a: int, b: int, borrow_in: int) -> tuple:
        r, b = self.forward_batch(mx.array([a]), mx.array([b]), mx.array([borrow_in]))
        return int(mx.argmax(r[0]).item()), int(mx.argmax(b[0]).item())


class LearnedMultTable(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.digit_embed = nn.Embedding(10, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 82)
    
    def forward_batch(self, a, b):
        emb = mx.concatenate([self.digit_embed(a), self.digit_embed(b)], axis=-1)
        return self.output(nn.relu(self.fc2(nn.relu(self.fc1(emb)))))
    
    def multiply(self, a: int, b: int) -> int:
        return int(mx.argmax(self.forward_batch(mx.array([a]), mx.array([b]))[0]).item())


class LearnedDivTable(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.dividend_embed = nn.Embedding(100, hidden_dim)
        self.divisor_embed = nn.Embedding(10, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.quotient_head = nn.Linear(hidden_dim, 100)
        self.remainder_head = nn.Linear(hidden_dim, 10)
    
    def forward_batch(self, dividend, divisor):
        emb = mx.concatenate([self.dividend_embed(dividend), self.divisor_embed(divisor)], axis=-1)
        h = nn.relu(self.fc2(nn.relu(self.fc1(emb))))
        return self.quotient_head(h), self.remainder_head(h)
    
    def divide(self, dividend: int, divisor: int) -> tuple:
        if divisor == 0:
            return 0, 0
        d = mx.array([min(dividend, 99)], dtype=mx.int32)
        v = mx.array([min(divisor, 9)], dtype=mx.int32)
        ql, rl = self.forward_batch(d, v)
        return int(mx.argmax(ql[0]).item()), int(mx.argmax(rl[0]).item())


class MathEngine(nn.Module):
    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        self.carry_table = LearnedCarryTable(hidden_dim)
        self.borrow_table = LearnedBorrowTable(hidden_dim)
        self.mult_table = LearnedMultTable(hidden_dim)
        self.div_table = LearnedDivTable(hidden_dim)
    
    def parse_number(self, s: str) -> list:
        return [int(c) for c in s.strip()][::-1]
    
    def digits_to_string(self, digits: list) -> str:
        result = digits[::-1]
        while len(result) > 1 and result[0] == 0:
            result = result[1:]
        return "".join(str(d) for d in result)
    
    def add(self, a: str, b: str) -> str:
        d1, d2 = self.parse_number(a), self.parse_number(b)
        max_len = max(len(d1), len(d2))
        d1 += [0] * (max_len - len(d1))
        d2 += [0] * (max_len - len(d2))
        result, carry = [], 0
        for i in range(max_len):
            r, carry = self.carry_table.add_with_carry(d1[i], d2[i], carry)
            result.append(r)
        if carry > 0:
            result.append(carry)
        return self.digits_to_string(result)
    
    def subtract(self, a: str, b: str) -> str:
        d1, d2 = self.parse_number(a), self.parse_number(b)
        max_len = max(len(d1), len(d2))
        d1 += [0] * (max_len - len(d1))
        d2 += [0] * (max_len - len(d2))
        result, borrow = [], 0
        for i in range(max_len):
            r, borrow = self.borrow_table.sub_with_borrow(d1[i], d2[i], borrow)
            result.append(r)
        return self.digits_to_string(result)
    
    def multiply(self, a: str, b: str) -> str:
        d2 = self.parse_number(b)
        total = "0"
        for i, digit in enumerate(d2):
            digits = self.parse_number(a)
            partial_result, carry = [], 0
            for d in digits:
                product = self.mult_table.multiply(d, digit)
                t = product + carry
                partial_result.append(t % 10)
                carry = t // 10
            while carry > 0:
                partial_result.append(carry % 10)
                carry //= 10
            partial = self.digits_to_string(partial_result) + "0" * i
            total = self.add(total, partial)
        return total
    
    def divide(self, a: str, b: str) -> Tuple[str, str]:
        if b == "0":
            return "0", "0"
        divisor = int(b)
        if divisor > 9:
            dividend = int(a)
            return str(dividend // divisor), str(dividend % divisor)
        digits = [int(c) for c in a]
        quotient, remainder = [], 0
        for d in digits:
            current = remainder * 10 + d
            q, remainder = self.div_table.divide(current, divisor)
            quotient.append(q)
        while len(quotient) > 1 and quotient[0] == 0:
            quotient = quotient[1:]
        return "".join(str(d) for d in quotient), str(remainder)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ============================================================
# LANGUAGE MODEL
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))
    
    def __call__(self, x):
        return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6) * self.weight


class SimpleMamba(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inner = dim * 2
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, kernel_size=4, padding=3, groups=inner)
        self.dt_proj = nn.Linear(inner, inner, bias=True)
        self.D = mx.ones((inner,))
        self.out_proj = nn.Linear(inner, dim, bias=False)
    
    def __call__(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = mx.split(xz, 2, axis=-1)
        x_conv = nn.silu(self.conv1d(x_in)[:, :L, :])
        dt = nn.softplus(self.dt_proj(x_conv))
        y = x_conv * mx.sigmoid(dt) + x_conv * self.D
        return self.out_proj(y * nn.silu(z))


class LanguageBase(nn.Module):
    def __init__(self, dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(256, dim)
        self.blocks = [{'norm': RMSNorm(dim), 'mamba': SimpleMamba(dim)} for _ in range(num_layers)]
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, 256, bias=False)
    
    def __call__(self, x, lora=None):
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            h = h + block['mamba'](block['norm'](h))
            if lora:
                h = lora.inject(h, i)
        return self.output(self.norm(h))
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, num_layers: int, rank: int = 8):
        super().__init__()
        self.scale = 16.0 / rank
        self.down = [nn.Linear(dim, rank, bias=False) for _ in range(num_layers)]
        self.up = [nn.Linear(rank, dim, bias=False) for _ in range(num_layers)]
    
    def inject(self, h, layer_idx):
        return h + self.up[layer_idx](self.down[layer_idx](h)) * self.scale


# ============================================================
# TRAINING WITH RETRY
# ============================================================

CARRY_DATA = [(a, b, c, (a+b+c) % 10, 1 if a+b+c >= 10 else 0) 
              for a in range(10) for b in range(10) for c in range(2)]

BORROW_DATA = []
for a in range(10):
    for b in range(10):
        for bo_in in range(2):
            val = a - b - bo_in
            if val < 0:
                BORROW_DATA.append((a, b, bo_in, val + 10, 1))
            else:
                BORROW_DATA.append((a, b, bo_in, val, 0))

MULT_DATA = [(a, b, a * b) for a in range(10) for b in range(10)]
DIV_DATA = [(d, v, d // v, d % v) for d in range(100) for v in range(1, 10)]


def train_table_until_100(math_engine, table_name, data, forward_fn, max_attempts=5):
    """Train a table until 100% accuracy"""
    for attempt in range(max_attempts):
        steps = 5000 * (attempt + 1)
        lr = 1e-2 / (attempt + 1)
        
        table = getattr(math_engine, table_name)
        opt = optim.AdamW(learning_rate=lr)
        
        for step in range(steps):
            batch = random.choices(data, k=64)
            
            def loss_fn(params):
                table.update(params)
                return forward_fn(table, batch)
            
            loss, grads = mx.value_and_grad(loss_fn)(table.trainable_parameters())
            opt.update(table, grads)
            mx.eval(table.parameters())
        
        # Test accuracy
        correct = test_table(math_engine, table_name, data)
        accuracy = correct / len(data)
        print(f"      {table_name}: {correct}/{len(data)} ({accuracy*100:.0f}%)")
        
        if correct == len(data):
            return True
    
    return False


def test_table(math_engine, table_name, data):
    if table_name == "carry_table":
        return sum(1 for a,b,c,r,co in data 
                   if math_engine.carry_table.add_with_carry(a,b,c) == (r,co))
    elif table_name == "borrow_table":
        return sum(1 for a,b,bo,r,boo in data 
                   if math_engine.borrow_table.sub_with_borrow(a,b,bo) == (r,boo))
    elif table_name == "mult_table":
        return sum(1 for a,b,p in data 
                   if math_engine.mult_table.multiply(a,b) == p)
    elif table_name == "div_table":
        return sum(1 for d,v,q,r in data 
                   if math_engine.div_table.divide(d,v) == (q,r))
    return 0


def carry_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    c = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    ct = mx.array([x[4] for x in batch], dtype=mx.int32)
    rl, cl = table.forward_batch(a, b, c)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(cl, axis=-1)+1e-10), ct[:, None], axis=1))
    return l1 + l2


def borrow_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    bo = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    bt = mx.array([x[4] for x in batch], dtype=mx.int32)
    rl, bl = table.forward_batch(a, b, bo)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(bl, axis=-1)+1e-10), bt[:, None], axis=1))
    return l1 + l2


def mult_forward(table, batch):
    a = mx.array([x[0] for x in batch], dtype=mx.int32)
    b = mx.array([x[1] for x in batch], dtype=mx.int32)
    targets = mx.array([x[2] for x in batch], dtype=mx.int32)
    logits = table.forward_batch(a, b)
    return -mx.mean(mx.take_along_axis(mx.log(mx.softmax(logits, axis=-1)+1e-10), targets[:, None], axis=1))


def div_forward(table, batch):
    d = mx.array([x[0] for x in batch], dtype=mx.int32)
    v = mx.array([x[1] for x in batch], dtype=mx.int32)
    qt = mx.array([x[2] for x in batch], dtype=mx.int32)
    rt = mx.array([x[3] for x in batch], dtype=mx.int32)
    ql, rl = table.forward_batch(d, v)
    l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(ql, axis=-1)+1e-10), qt[:, None], axis=1))
    l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
    return l1 + l2


def train_all_math_until_100(math_engine):
    """Train all math tables until 100%"""
    print("   Training carry table...")
    train_table_until_100(math_engine, "carry_table", CARRY_DATA, carry_forward)
    
    print("   Training borrow table...")
    train_table_until_100(math_engine, "borrow_table", BORROW_DATA, borrow_forward)
    
    print("   Training mult table...")
    train_table_until_100(math_engine, "mult_table", MULT_DATA, mult_forward)
    
    print("   Training div table...")
    train_table_until_100(math_engine, "div_table", DIV_DATA, div_forward)


def encode(text, max_len=48):
    tokens = [ord(c) for c in text[:max_len]]
    return tokens + [0] * (max_len - len(tokens))


def train_lora(base, lora, data, steps=5000):
    """Train LoRA with generated data"""
    opt = optim.AdamW(learning_rate=1e-3)
    
    for step in range(steps):
        batch = random.choices(data, k=32)
        inputs, targets = [], []
        for q, a in batch:
            text = f"{q}{a}"
            tokens = encode(text)
            inputs.append(tokens[:-1])
            targets.append(tokens[1:])
        
        x = mx.array(inputs, dtype=mx.int32)
        y = mx.array(targets, dtype=mx.int32)
        
        def loss_fn(params):
            lora.update(params)
            logits = base(x, lora)
            logits_flat = logits.reshape(-1, 256)
            targets_flat = y.reshape(-1)
            log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
            return -mx.mean(mx.take_along_axis(log_probs, targets_flat[:, None], axis=1))
        
        loss, grads = mx.value_and_grad(loss_fn)(lora.trainable_parameters())
        opt.update(lora, grads)
        mx.eval(lora.parameters())
        
        if step % 1000 == 0:
            print(f"      Step {step}: loss={float(loss):.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("     GHOST v20 IMPROVED - Generated Data + Retry Training")
    print("=" * 70)
    
    # Initialize
    math_engine = MathEngine(hidden_dim=48)
    base = LanguageBase(dim=128, num_layers=4)
    logic_lora = LoRAAdapter(128, 4, rank=16)
    code_lora = LoRAAdapter(128, 4, rank=16)
    fact_lora = LoRAAdapter(128, 4, rank=8)
    
    mx.eval(math_engine.parameters())
    mx.eval(base.parameters())
    
    print(f"\n📦 Model: {math_engine.count_params() + base.count_params():,} params")
    
    # Generate data
    print("\n📊 GENERATING DATA...")
    print("-" * 70)
    code_data = generate_code_data()
    logic_data = generate_logic_data()
    fact_data = generate_fact_data()
    
    start = time.perf_counter()
    
    # Train math until 100%
    print("\n📐 TRAINING MATH (until 100%)...")
    print("-" * 70)
    train_all_math_until_100(math_engine)
    
    # Train LoRAs with generated data
    print("\n📚 TRAINING LoRAs (with generated data)...")
    print("-" * 70)
    
    print(f"   Logic LoRA ({len(logic_data)} examples):")
    train_lora(base, logic_lora, logic_data, steps=5000)
    
    print(f"\n   Code LoRA ({len(code_data)} examples):")
    train_lora(base, code_lora, code_data, steps=5000)
    
    print(f"\n   Fact LoRA ({len(fact_data)} examples):")
    train_lora(base, fact_lora, fact_data, steps=3000)
    
    train_time = time.perf_counter() - start
    
    # Test
    print("\n" + "=" * 70)
    print("              TESTING")
    print("=" * 70)
    
    print("\n➕ MATH:")
    math_tests = [("999", "1", "+"), ("12345", "67890", "+"), 
                  ("1000", "1", "-"), ("100", "100", "*"), 
                  ("100", "7", "/")]
    
    for a, b, op in math_tests:
        if op == "+":
            result = math_engine.add(a, b)
            expected = str(int(a) + int(b))
        elif op == "-":
            result = math_engine.subtract(a, b)
            expected = str(int(a) - int(b))
        elif op == "*":
            result = math_engine.multiply(a, b)
            expected = str(int(a) * int(b))
        elif op == "/":
            q, r = math_engine.divide(a, b)
            result = f"{q} R {r}"
            eq, er = int(a) // int(b), int(a) % int(b)
            expected = f"{eq} R {er}"
        
        status = "✅" if result == expected else "❌"
        print(f"   {a} {op} {b} = {result} {status}")
    
    print(f"\n⏱️ Total training: {train_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
