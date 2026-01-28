# Ghost v20: Complete Integrated AI System

**100% Math Accuracy + Auto-Generated Training Data**

## 🎯 Key Features

| Component | Accuracy | Examples |
|-----------|----------|----------|
| **Math** (Add/Sub/Mul/Div) | **100%** | Infinite (any size) |
| Logic LoRA | High | 1804 generated |
| Code LoRA | Good | 300 generated |
| Fact LoRA | High | 88 country facts |

## 📊 Training Results

```
Math Tables (all 100%):
├── Carry:  200/200 ✅
├── Borrow: 200/200 ✅
├── Mult:   100/100 ✅
└── Div:    900/900 ✅ (with retry)

Test Results:
├── 999 + 1 = 1000 ✅
├── 12345 + 67890 = 80235 ✅
├── 1000 - 1 = 999 ✅
├── 100 * 100 = 10000 ✅
└── 100 / 7 = 14 R 2 ✅
```

## 🚀 Quick Start

```bash
# Standard version (with save/load)
python ghost_model_v20/ghost_v20.py

# Improved version (generated data + retry)
python ghost_model_v20/ghost_v20_improved.py
```

## 📁 Files

| File | Purpose |
|------|---------|
| `ghost_v20.py` | Main with save/load |
| `ghost_v20_improved.py` | Retry training + generators |
| `data/generators.py` | 3000+ training examples |
| `weights/` | Pre-trained checkpoints |

## 🔧 Architecture (811K params)

```
GHOST V20
├── Math Engine (82K)
│   ├── LearnedCarryTable
│   ├── LearnedBorrowTable
│   ├── LearnedMultTable
│   └── LearnedDivTable
│
└── Language Model (729K)
    ├── Mamba Blocks (4 layers)
    └── LoRA Adapters
        ├── Logic (rank=16)
        ├── Code (rank=16)
        └── Fact (rank=8)
```

## 🆕 Improvements

1. **Retry Training** - Tables train until 100%
2. **Data Generators** - 3000+ auto-generated examples
3. **Save/Load** - Instant restart after first training
4. **Type Detection** - Auto-routes to correct specialist
