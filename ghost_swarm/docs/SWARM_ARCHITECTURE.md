# Ghost Swarm Architecture

**Concept**: A local "hive mind" of specialized Ghost v6 agents running in parallel on a single consumer device.

---

## 1. Core Principles

### A. Specialization > Generalization
Instead of one massive model that knows everything, we train **Micro-Agents** optimized for specific tasks:
- **GitAgent**: Knows `git`, diffs, commits, branches.
- **PythonAgent**: Knows syntax, standard lib, pandas.
- **LogAgent**: Knows regex, error patterns, timestamps.

### B. Shared Knowledge Lake (Vector DB)
Agents don't have large internal memory. They share a **single persistent memory store**:
- Agent A learns a fact → Writes to Shared Memory
- Agent B queries Shared Memory → Instantly knows the fact

### C. The Dispatcher (Orchestrator)
A lightweight router that decides:
1. "User typed a command" → Send to **ShellAgent**
2. "ShellAgent failed" → Dispatch to **DebugAgent**
3. "User wants to code" → Wake up **PythonAgent**

---

## 2. Directory Structure

```
ghost_swarm/
├── core/
│   └── ghost_v6.py       # The brain (6.5M params)
├── runtime/
│   ├── dispatcher.py     # Task router
│   ├── memory.py         # Shared vector DB
│   └── agent_pool.py     # Process manager
├── roles/
│   ├── base_agent.py     # Base class
│   ├── coder_agent.py    # Python/JS specialist
│   └── ops_agent.py      # Bash/Docker specialist
└── experiments/
    └── spawn_test.py     # 100-agent capacity test
```

---

## 3. Communication Protocol (Inter-Process)

Agents communicate via lightweight **ZeroMQ** or **Python Multiprocessing Queues**:

**Message Format:**
```json
{
  "from": "GitAgent_01",
  "to": "Dispatcher",
  "intent": "commit_suggestion",
  "content": "git commit -m 'Fix typo in README'",
  "confidence": 0.98
}
```

---

## 4. Resource Allocation

| Role | Count | Precision | RAM per Unit | Total RAM |
|:---|:---|:---|:---|:---|
| **Dispatcher** | 1 | float32 | 10 MB | 10 MB |
| **Active Code Agents** | 2 | 4-bit | 6 MB | 12 MB |
| **Idle Monitors** | 50 | 4-bit (swap) | 1 MB | 50 MB |
| **Background Learners** | 5 | float16 | 15 MB | 75 MB |
| **Transformation** | - | - | - | **~150 MB** |

**Total Footprint**: < 200 MB for a fully functional Swarm.

---

## 5. Implementation Roadmap

### Phase 1: The Runtime
- [ ] Build `Dispatcher` class
- [ ] Implement `SharedMemory` interface
- [ ] Create `spawn_test.py` to verify RAM usage

### Phase 2: The Specialists
- [ ] Train **OpsAgent** (on Bash data)
- [ ] Train **CoderAgent** (on Python data)
- [ ] Verify specialization improves accuracy

### Phase 3: The Hive
- [ ] Connect agents via message bus
- [ ] Test multi-agent problem solving (e.g., Code Agent writes file → Ops Agent runs it → Debug Agent fixes error)
