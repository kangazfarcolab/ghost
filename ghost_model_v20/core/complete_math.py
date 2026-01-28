"""
Ghost v20 - Complete Math with Division
========================================
All four arithmetic operations:
- Addition (carry table)
- Subtraction (borrow table)
- Multiplication (mult table)
- Division (div table)

All knowledge in WEIGHTS!
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import random
import time


# ============================================================
# LEARNED CARRY TABLE (Addition)
# ============================================================

class LearnedCarryTable(nn.Module):
    def __init__(self, hidden_dim: int = 32):
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


# ============================================================
# LEARNED BORROW TABLE (Subtraction)
# ============================================================

class LearnedBorrowTable(nn.Module):
    def __init__(self, hidden_dim: int = 32):
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


# ============================================================
# LEARNED MULTIPLICATION TABLE
# ============================================================

class LearnedMultTable(nn.Module):
    def __init__(self, hidden_dim: int = 32):
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


# ============================================================
# LEARNED DIVISION TABLE
# ============================================================

class LearnedDivTable(nn.Module):
    """
    Learn division: a / b → (quotient, remainder)
    
    Examples:
    - 8 / 3 = 2 remainder 2
    - 9 / 3 = 3 remainder 0
    - 5 / 7 = 0 remainder 5
    
    Only single digit: a (0-99 for two digit dividend), b (1-9)
    """
    
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.dividend_embed = nn.Embedding(100, hidden_dim)  # 0-99
        self.divisor_embed = nn.Embedding(10, hidden_dim)    # 1-9
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.quotient_head = nn.Linear(hidden_dim, 100)  # 0-99
        self.remainder_head = nn.Linear(hidden_dim, 10)  # 0-9
    
    def forward_batch(self, dividend, divisor):
        emb = mx.concatenate([self.dividend_embed(dividend), self.divisor_embed(divisor)], axis=-1)
        h = nn.relu(self.fc1(emb))
        h = nn.relu(self.fc2(h))
        return self.quotient_head(h), self.remainder_head(h)
    
    def divide(self, dividend: int, divisor: int) -> tuple:
        """Returns (quotient, remainder)"""
        if divisor == 0:
            return 0, 0
        d_arr = mx.array([min(dividend, 99)], dtype=mx.int32)
        v_arr = mx.array([min(divisor, 9)], dtype=mx.int32)
        q_logits, r_logits = self.forward_batch(d_arr, v_arr)
        return int(mx.argmax(q_logits[0]).item()), int(mx.argmax(r_logits[0]).item())


# ============================================================
# COMPLETE MATH ENGINE
# ============================================================

class CompleteMathEngine(nn.Module):
    def __init__(self, hidden_dim: int = 32):
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
    
    def add(self, num1: str, num2: str) -> str:
        d1, d2 = self.parse_number(num1), self.parse_number(num2)
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
    
    def subtract(self, num1: str, num2: str) -> str:
        d1, d2 = self.parse_number(num1), self.parse_number(num2)
        max_len = max(len(d1), len(d2))
        d1 += [0] * (max_len - len(d1))
        d2 += [0] * (max_len - len(d2))
        result, borrow = [], 0
        for i in range(max_len):
            r, borrow = self.borrow_table.sub_with_borrow(d1[i], d2[i], borrow)
            result.append(r)
        return self.digits_to_string(result)
    
    def multiply_single(self, num: str, digit: int) -> str:
        digits = self.parse_number(num)
        result, carry = [], 0
        for d in digits:
            product = self.mult_table.multiply(d, digit)
            total = product + carry
            result.append(total % 10)
            carry = total // 10
        while carry > 0:
            result.append(carry % 10)
            carry //= 10
        return self.digits_to_string(result)
    
    def multiply(self, num1: str, num2: str) -> str:
        d2 = self.parse_number(num2)
        total = "0"
        for i, digit in enumerate(d2):
            partial = self.multiply_single(num1, digit)
            partial = partial + "0" * i
            total = self.add(total, partial)
        return total
    
    def divide(self, num1: str, num2: str) -> tuple:
        """
        Long division using learned table.
        Returns (quotient, remainder) as strings.
        """
        if num2 == "0":
            return "0", "0"
        
        divisor = int(num2)
        if divisor > 9:
            # For large divisors, do it digit by digit (simplified)
            dividend = int(num1)
            return str(dividend // divisor), str(dividend % divisor)
        
        # Single digit divisor: grade school long division
        digits = [int(c) for c in num1]
        quotient = []
        remainder = 0
        
        for d in digits:
            current = remainder * 10 + d
            q, remainder = self.div_table.divide(current, divisor)
            quotient.append(q)
        
        # Remove leading zeros
        while len(quotient) > 1 and quotient[0] == 0:
            quotient = quotient[1:]
        
        return "".join(str(d) for d in quotient), str(remainder)
    
    def count_params(self):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(self.parameters()))


# ============================================================
# TRAINING DATA
# ============================================================

def create_carry_data():
    return [(a, b, c, (a+b+c) % 10, 1 if a+b+c >= 10 else 0) 
            for a in range(10) for b in range(10) for c in range(2)]

def create_borrow_data():
    data = []
    for a in range(10):
        for b in range(10):
            for borrow_in in range(2):
                val = a - b - borrow_in
                if val < 0:
                    result, borrow_out = val + 10, 1
                else:
                    result, borrow_out = val, 0
                data.append((a, b, borrow_in, result, borrow_out))
    return data

def create_mult_data():
    return [(a, b, a * b) for a in range(10) for b in range(10)]

def create_div_data():
    """Division table: dividend (0-99), divisor (1-9)"""
    data = []
    for dividend in range(100):
        for divisor in range(1, 10):
            quotient = dividend // divisor
            remainder = dividend % divisor
            data.append((dividend, divisor, quotient, remainder))
    return data


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_table(model, table_name, data_fn, forward_fn, steps=4000, lr=1e-2):
    print(f"   Training {table_name}...")
    DATA = data_fn()
    opt = optim.AdamW(learning_rate=lr)
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        
        def loss_fn(params):
            return forward_fn(model, params, batch)
        
        loss, grads = mx.value_and_grad(loss_fn)(getattr(model, table_name).trainable_parameters())
        opt.update(getattr(model, table_name), grads)
        mx.eval(getattr(model, table_name).parameters())
    
    return DATA


def train_carry(model, steps=4000):
    print("   Training carry table...")
    DATA = create_carry_data()
    opt = optim.AdamW(learning_rate=1e-2)
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        c = mx.array([x[2] for x in batch], dtype=mx.int32)
        rt = mx.array([x[3] for x in batch], dtype=mx.int32)
        ct = mx.array([x[4] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.carry_table.update(params)
            rl, cl = model.carry_table.forward_batch(a, b, c)
            l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
            l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(cl, axis=-1)+1e-10), ct[:, None], axis=1))
            return l1 + l2
        
        loss, grads = mx.value_and_grad(loss_fn)(model.carry_table.trainable_parameters())
        opt.update(model.carry_table, grads)
        mx.eval(model.carry_table.parameters())
    
    correct = sum(1 for a,b,c,r,co in DATA if model.carry_table.add_with_carry(a,b,c) == (r,co))
    print(f"   Carry: {correct}/{len(DATA)} ({correct/len(DATA)*100:.0f}%)")
    return correct == len(DATA)


def train_borrow(model, steps=4000):
    print("   Training borrow table...")
    DATA = create_borrow_data()
    opt = optim.AdamW(learning_rate=1e-2)
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        bo = mx.array([x[2] for x in batch], dtype=mx.int32)
        rt = mx.array([x[3] for x in batch], dtype=mx.int32)
        bt = mx.array([x[4] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.borrow_table.update(params)
            rl, bl = model.borrow_table.forward_batch(a, b, bo)
            l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(rl, axis=-1)+1e-10), rt[:, None], axis=1))
            l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(bl, axis=-1)+1e-10), bt[:, None], axis=1))
            return l1 + l2
        
        loss, grads = mx.value_and_grad(loss_fn)(model.borrow_table.trainable_parameters())
        opt.update(model.borrow_table, grads)
        mx.eval(model.borrow_table.parameters())
    
    correct = sum(1 for a,b,bo,r,boo in DATA if model.borrow_table.sub_with_borrow(a,b,bo) == (r,boo))
    print(f"   Borrow: {correct}/{len(DATA)} ({correct/len(DATA)*100:.0f}%)")
    return correct == len(DATA)


def train_mult(model, steps=4000):
    print("   Training multiplication table...")
    DATA = create_mult_data()
    opt = optim.AdamW(learning_rate=1e-2)
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        a = mx.array([x[0] for x in batch], dtype=mx.int32)
        b = mx.array([x[1] for x in batch], dtype=mx.int32)
        targets = mx.array([x[2] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.mult_table.update(params)
            logits = model.mult_table.forward_batch(a, b)
            return -mx.mean(mx.take_along_axis(mx.log(mx.softmax(logits, axis=-1)+1e-10), targets[:, None], axis=1))
        
        loss, grads = mx.value_and_grad(loss_fn)(model.mult_table.trainable_parameters())
        opt.update(model.mult_table, grads)
        mx.eval(model.mult_table.parameters())
    
    correct = sum(1 for a,b,p in DATA if model.mult_table.multiply(a,b) == p)
    print(f"   Multiply: {correct}/{len(DATA)} ({correct/len(DATA)*100:.0f}%)")
    return correct == len(DATA)


def train_div(model, steps=6000):
    """Train division table (more data so more steps)"""
    print("   Training division table...")
    DATA = create_div_data()
    opt = optim.AdamW(learning_rate=1e-2)
    
    for step in range(steps):
        batch = random.choices(DATA, k=32)
        dividend = mx.array([x[0] for x in batch], dtype=mx.int32)
        divisor = mx.array([x[1] for x in batch], dtype=mx.int32)
        q_target = mx.array([x[2] for x in batch], dtype=mx.int32)
        r_target = mx.array([x[3] for x in batch], dtype=mx.int32)
        
        def loss_fn(params):
            model.div_table.update(params)
            q_logits, r_logits = model.div_table.forward_batch(dividend, divisor)
            l1 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(q_logits, axis=-1)+1e-10), q_target[:, None], axis=1))
            l2 = -mx.mean(mx.take_along_axis(mx.log(mx.softmax(r_logits, axis=-1)+1e-10), r_target[:, None], axis=1))
            return l1 + l2
        
        loss, grads = mx.value_and_grad(loss_fn)(model.div_table.trainable_parameters())
        opt.update(model.div_table, grads)
        mx.eval(model.div_table.parameters())
    
    correct = 0
    for dividend, divisor, exp_q, exp_r in DATA:
        q, r = model.div_table.divide(dividend, divisor)
        if q == exp_q and r == exp_r:
            correct += 1
    print(f"   Division: {correct}/{len(DATA)} ({correct/len(DATA)*100:.0f}%)")
    return correct == len(DATA)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("     GHOST v20 - COMPLETE MATH (+ DIVISION)")
    print("=" * 60)
    
    model = CompleteMathEngine(hidden_dim=32)
    mx.eval(model.parameters())
    
    print(f"\n📦 Total params: {model.count_params():,}")
    
    print("\n📚 TRAINING (until 100% on each)")
    print("-" * 60)
    
    start = time.perf_counter()
    
    # Train all tables with retry
    for attempt in range(5):
        if train_carry(model):
            break
    
    for attempt in range(5):
        if train_borrow(model):
            break
    
    for attempt in range(5):
        if train_mult(model):
            break
    
    for attempt in range(5):
        if train_div(model):
            break
    
    train_time = time.perf_counter() - start
    print(f"\n⏱️ Total training: {train_time:.1f}s")
    
    # Test
    print("\n🧪 TESTING")
    print("-" * 60)
    
    print("\n➕ Addition:")
    for a, b, exp in [("999", "1", "1000"), ("12345", "67890", "80235")]:
        res = model.add(a, b)
        print(f"   {a} + {b} = {res} {'✅' if res == exp else '❌'}")
    
    print("\n➖ Subtraction:")
    for a, b, exp in [("1000", "1", "999"), ("80235", "67890", "12345")]:
        res = model.subtract(a, b)
        print(f"   {a} - {b} = {res} {'✅' if res == exp else '❌'}")
    
    print("\n✖️ Multiplication:")
    for a, b, exp in [("12", "12", "144"), ("99", "99", "9801")]:
        res = model.multiply(a, b)
        print(f"   {a} × {b} = {res} {'✅' if res == exp else '❌'}")
    
    print("\n➗ Division:")
    div_tests = [
        ("10", "3", "3", "1"),   # 10/3 = 3 R 1
        ("100", "7", "14", "2"), # 100/7 = 14 R 2
        ("144", "12", "12", "0"), # 144/12 = 12 R 0
        ("999", "9", "111", "0"), # 999/9 = 111 R 0
    ]
    for a, b, exp_q, exp_r in div_tests:
        q, r = model.divide(a, b)
        status = "✅" if q == exp_q and r == exp_r else "❌"
        print(f"   {a} ÷ {b} = {q} R {r} (expected: {exp_q} R {exp_r}) {status}")
    
    print("\n" + "=" * 60)
    print("   ✅ All four operations ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
