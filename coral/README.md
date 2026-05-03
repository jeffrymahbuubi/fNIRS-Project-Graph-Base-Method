# CORAL Setup — fNIRS GATv2 Optimization

## Overview

This directory contains a [CORAL](https://github.com/Human-Agent-Society/CORAL) task for autonomously optimizing the GATv2Conv-based graph neural network for fNIRS anxiety classification.

CORAL spawns autonomous Claude Code agents that iteratively modify the codebase, run graded evaluations, and share knowledge to improve the validation F1 score.

---

## Directory Structure

```
coral/
├── README.md                  ← this file
└── fnirs_gat/
    ├── task.yaml              ← CORAL task configuration
    ├── eval/
    │   └── grader.py          ← TaskGrader: runs solution.run(), returns holdout F1
    ├── results/               ← CORAL runtime output (gitignored)
    │   └── fnirs-gat-hbo-optimization/
    │       └── <timestamp>/
    │           ├── repo/      ← CORAL's own git repo (separate from project repo)
    │           ├── agents/    ← git worktrees for each agent
    │           └── .coral/    ← shared state: attempts, notes, skills
    └── seed/
        ├── .gitignore
        ├── solution.py        ← baseline config; agents modify this
        └── core/              ← full copy of src/core/ (agents may also modify this)
            ├── __init__.py
            ├── config.py      ← ExperimentConfig dataclass
            ├── dataset.py     ← fNIRSGraphDataset + loaders
            ├── models.py      ← FlexibleGATNet (GATv2Conv + optional GINEConv)
            ├── training.py    ← train/eval loops, perform_holdout_training
            ├── transforms.py  ← StandardizeGraphFeatures + augmentation
            └── utils.py       ← statistical features, correlation, coherence
```

> **Git isolation**: `coral/fnirs_gat/results/` is listed in `.gitignore`. All agent commits happen inside CORAL's own nested git repo at `results/.../repo/` — the main project repo is never touched.

---

## Task Configuration (`fnirs_gat/task.yaml`)

| Field | Value |
|---|---|
| Task name | `fnirs-gat-hbo-optimization` |
| Agents | **3** parallel Claude Sonnet agents |
| Max turns per restart | 100 |
| Grader timeout | 360 seconds |
| Score direction | maximize (higher F1 is better) |
| Results directory | `fnirs_gat/results/` (absolute path) |

### Fixed Parameters (agents cannot change these)

| Parameter | Value | Reason |
|---|---|---|
| `data_type` | `hbo` | Best signal (F1 0.8065 vs HBT 0.7479, HBR 0.7376) |
| `max_trials` | `2` | mt2 beats mt4 by 5–12pp across all signals |
| `task_type` | `GNG` | Cognitive task subdirectory |
| `validation` | `holdout` | Fast eval (~60-90s vs 5× for kfold) |
| `val_ratio` | `0.2` | Consistent with prior 5-fold subject-level splits |
| `random_state` | `42` | Reproducible holdout splits |
| `directed` | `True` | Directed edges (best config from sweep) |
| `self_loops` | `True` | Self-loops enabled (best config from sweep) |

### What Agents Optimize

Ordered by expected impact (highest first):

1. **Augmentation** — completely untested in the sweep:
   - `augment`, `edge_dropout_p`, `feature_mask_p`, `feature_mask_mode`, `use_rwpe`, `rwpe_walk_length`

2. **Loss function** — untested on the 29/33 class imbalance:
   - `use_focal_loss`, `focal_alpha`, `focal_gamma`
   - `use_class_weights`, `sqrt_class_weights`

3. **Training schedule**:
   - `lr`, `epochs`, `patience`, `batch_size`

4. **Architecture**:
   - `n_layers`, `n_filters`, `n_heads`, `fc_size`, `dropout`
   - `use_norm`, `norm_type`, `use_gine_first_layer`, `use_residual`

5. **Model code** — agents may modify `core/models.py` to add new layer types or pooling strategies.

### Agent Exploration Strategy

Agents are instructed (via `task.yaml` tips) to:

- **Test generalizability**: when an improvement is found (e.g., augmentation), verify it also holds with a simpler architecture (`n_filters=64, n_heads=4`). Robust findings hold on both; architecture-specific findings are logged in notes.
- **Explore architecture sensitivity**: the prior sweep only varied `data_type × max_trials` — the current architecture was never hyperparameter-searched. Agents are permitted to explore different filter/head combinations in combination with the best regularization settings they discover.

---

## Baseline Performance

From `research/experiments/20260429/RESULTS_SUMMARY.md` — 5-fold stratified CV, subject-level splits, seed 42:

| Dataset | Signal | mt | Mean Acc | Mean F1 |
|---|---|---|---|---|
| With additional data (62 subjects) | HBO | 2 | **0.8237** | **0.8065** |
| With additional data (62 subjects) | HBT | 2 | 0.7590 | 0.7479 |
| With additional data (62 subjects) | HBR | 2 | 0.7327 | 0.7376 |

The CORAL optimization targets this baseline: **holdout F1 > 0.8065**.

### Baseline Model Config

```
n_layers=2, n_filters=[112, 32], n_heads=[6, 4], fc_size=96
dropout=0.4, use_residual=True, use_norm=True (batch), use_gine_first_layer=True
epochs=100, patience=9999, lr=1e-3, batch_size=8
augment=False, use_focal_loss=False, use_class_weights=False
```

---

## How It Works

### Grader Flow

```
agent modifies solution.py
    → agent runs: uv run coral eval -m "description"
    → CORAL stages + commits the change
    → grader.py runs in a subprocess
    → grader spawns: /path/to/.venv/bin/python -c "import solution; print(json.dumps({'f1_score': solution.run(data_dir)}))"
    → grader reads F1 from stdout JSON
    → CORAL records score + feedback to .coral/attempts/
    → agent receives score and decides next move
```

### Key Implementation Details

- **Python executable**: The grader uses the project venv at `src/.venv/bin/python` to ensure torch, torch_geometric, and all dependencies are available.
- **Data directory**: The grader passes `data/processed-new` (62 subjects: 29 anxiety / 33 healthy) as an absolute path.
- **Temp directories**: `solution.py` uses `tempfile.mkdtemp()` for `exp_dir` so each eval writes to a throwaway location.
- **Score key**: `perform_holdout_training` returns a dict; the grader reads `results["f1_score"]`.
- **Seed repo**: CORAL detects no `.git` in `seed/` and auto-initializes a fresh git repo, then copies `seed/` contents in. The main project repo is never affected.

---

## Running CORAL

### Starting a Run

CORAL must be started from its own directory. All paths in `task.yaml` are absolute.

```bash
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL

# Default: runs in tmux session in the background
uv run coral start -c /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat/task.yaml

# With verbose agent output streamed to terminal
uv run coral start -c /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat/task.yaml run.verbose=true

# With verbose output AND web dashboard simultaneously
uv run coral start -c /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat/task.yaml run.verbose=true run.ui=true

# Override agent count or model at launch
uv run coral start -c /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat/task.yaml agents.count=3 agents.model=opus
```

### Monitoring — IMPORTANT: CWD Issue

`coral ui`, `coral status`, `coral log` etc. find results by **walking up from the current working directory** looking for a `results/` folder. Running them from the CORAL library directory (`references/library/CORAL/`) finds CORAL's own examples — not this project's run.

**Always run monitoring commands from `coral/fnirs_gat/`:**

```bash
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat

uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral status
uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral log
uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral ui
uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral notes
uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral stop
uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral resume
```

**Recommended: add a shell alias** (run once, then source):

```bash
echo 'alias coral="uv run --project /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/references/library/CORAL coral"' >> ~/.bashrc
source ~/.bashrc
```

Then from `coral/fnirs_gat/`, all commands work without the long prefix:

```bash
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat
coral status
coral log
coral log --recent -n 10
coral log --search "augment"
coral notes
coral skills
coral show <commit-hash>
coral ui
```

### Viewing Live Agent Output

```bash
# Attach to the running tmux session (Ctrl+B then D to detach without stopping)
tmux attach -t coral-fnirs-gat-hbo-optimization-<timestamp>

# Find the session name if you forgot it
tmux ls
```

### Accessing the UI from a Remote Machine (Tailscale / SSH)

The CORAL UI server binds to `127.0.0.1:8420` (loopback only). Opening `http://127.0.0.1:8420/` in a browser on your **Mac** hits the Mac's own localhost — not the remote server — and will load indefinitely.

**Fix: SSH port forward from your Mac terminal**

```bash
ssh -L 8420:127.0.0.1:8420 user@<tailscale-ip>
# e.g.: ssh -L 8420:127.0.0.1:8420 user@100.115.115.16
```

Then open `http://127.0.0.1:8420/` in your Mac browser. The tunnel forwards Mac port 8420 → remote port 8420.

**If you get `bind: Address already in use`**, stale SSH tunnel processes are holding the port on your Mac. Either:

```bash
# Option A: kill the stale processes on Mac, then reconnect
lsof -ti:8420 | xargs kill -9

# Option B: use a different local port (no cleanup needed)
ssh -L 9090:127.0.0.1:8420 user@<tailscale-ip>
# then open http://127.0.0.1:9090/ instead
```

Stale tunnel processes often release on their own after ~30–60 seconds of being blocked — retrying after a short wait also works.

**Verify the backend is live** before debugging the UI:

```bash
curl http://127.0.0.1:8420/api/status
# Should return JSON with manager_alive, agents, best_score, etc.
```

### Stopping

```bash
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat
coral stop    # graceful shutdown (recommended)
# or: Ctrl+C in the tmux session (once = graceful, twice = force kill)
```

### Resuming

```bash
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method/coral/fnirs_gat
coral resume                                        # resume from last run
coral resume agents.model=opus                      # resume with a different model
coral resume -i "Focus on augmentation only"        # resume with extra instruction
```

---

## Stopping Criteria

CORAL **does not stop automatically** — it runs until manually stopped. Agent processes restart automatically after each `max_turns=100` limit; only `coral stop` ends the run.

| Signal | Action |
|---|---|
| F1 hasn't improved for 10+ consecutive evals | `coral stop` — search space likely exhausted |
| F1 exceeds your target (e.g., > 0.83) | `coral stop` — record best attempt, validate with 5-fold |
| After overnight run | Check `coral log` in the morning; stop if plateau |

**Cost estimate**: 3 agents × Sonnet × ~$0.03-0.10/turn → approximately **$30-90 for a 4-8 hour run**.

---

## After CORAL: Validating the Best Config

CORAL uses holdout validation (fast, single split). After CORAL identifies a best config, validate it properly with 5-fold CV using the main pipeline:

```bash
# 1. Copy best solution.py config values back to src/core/experiment_config.yaml
# 2. Run with kfold validation
cd /home/user/jeffrymahbuubi/PROJECTS/2-fNIRS-Graph-Base-Method

python -m src.core.main \
  --data_dir data/processed-new \
  --save_dir research/experiments/$(date +%Y%m%d)/coral-best \
  --task GNG \
  --data_type hbo \
  --max_trials 2 \
  --directed \
  --self_loops \
  --validation kfold \
  --k_folds 5 \
  --splits_json data/splits/kfold_splits_processed_new.json \
  --config src/core/experiment_config.yaml \
  --epochs 100
```

Results in `research/experiments/` follow the same structure as the 2026-04-29 sweep.

---

## Modifying the Search

To change what agents optimize, edit `fnirs_gat/task.yaml`:

- **Agent count**: `agents.count: 3` is current (higher cost, faster parallel exploration)
- **More turns**: `agents.max_turns: 200` (default CORAL value; more exploration per restart)
- **Smarter model**: `agents.model: opus` (higher quality reasoning, higher cost)
- **Literature review first**: add `agents.warmstart.enabled: true` + `agents.research: true`
- **Longer eval timeout**: increase `grader.timeout` if 100-epoch runs exceed 360s

To expand what agents can modify, update the `task.description` section in `task.yaml` — agents read this as their instruction set.
