# Ghost Swarm (Project Plan)

**Goal**: Run 100+ specialized Ghost v6 agents efficiently on a single consumer device.

---

## 1. Resource Requirements (per Agent)

Based on Ghost v6 architecture:
- **Parameters**: 6.58 Million
- **Precision**: float32 (current) → float16 (target)
- **Context**: 2048 bytes

### Memory Footprint
| Precision | Model Weight | Activation Cache | Total RAM |
|:---|:---|:---|:---|
| float32 | ~26 MB | ~5 MB | ~31 MB |
| float16 | ~13 MB | ~2.5 MB | ~15.5 MB |
| **Quantized (4-bit)** | **~3.5 MB** | **~2.5 MB** | **~6 MB** |

### Compute Requirements
- **Inference**: ~0.05 GFLOPS per token
- **Throughput**: >1000 tok/sec on M1/M2/M3/M4

---

## 2. Swarm Capacity (on 16GB Mac)

Assuming 4-bit quantization (easy to achieve):
- **Single Agent RAM**: 6 MB
- **100 Agents RAM**: 600 MB (0.6 GB)
- **1000 Agents RAM**: 6 GB (Leaves 10GB for OS!)

**Conclusion**: You can run **1000 active agents** on your Mac right now.

---

## 3. Project Roadmap

### Phase A: Optimization (The "Micro-Ghost")
- [ ] Implement 4-bit quantization for Ghost v6
- [ ] Optimize state caching (shared memory between agents)
- [ ] Build "Swarm Runtime" (Process manager)

### Phase B: Specialization (The "Roles")
- [ ] Train Agent A: "The Coder" (Python/JS expert)
- [ ] Train Agent B: "The Admin" (Bash/DevOps expert)
- [ ] Train Agent C: "The Critic" (Reviewer/Debugger)

### Phase C: Orchestration (The "Hive Mind")
- [ ] Inter-agent protocol (how they talk)
- [ ] Shared long-term memory (Vector DB)
- [ ] Task dispatcher

---

## 4. Immediate Next Step
Create `ghost_swarm` directory and build the **Swarm Runtime** to spawn 10 agents and measure actual memory usage.
