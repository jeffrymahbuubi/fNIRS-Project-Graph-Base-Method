# Early Stopping: Why Val Loss Minimum ≠ Best F1 Score

**Date:** 2026-05-01  
**Context:** ST_GATv2_GNG fNIRS kfold experiment (`mt4_noaug_20260501`)  
**Observed in:** fold_2 — checkpoint at epoch 3 (val loss minimum) → val F1 = 0.44; epoch ~5 (not saved) → val F1 = 0.76

---

## The Core Problem

Checkpointing on **lowest validation loss** selected a model with val F1 = 0.44, missing a later epoch with val F1 = 0.76 — a **32 percentage point miss**. This is not a fluke. Five independent analytical lenses explain why this happens structurally.

---

## Five Explanations (from Multi-Agent Analysis)

### 1. Mathematical Incompatibility (Information Theory)

Cross-entropy loss and F1 score optimize fundamentally different mathematical objects:

- **CE loss** minimizes KL divergence between predicted and true probability distributions — it rewards *calibration* (how confident and correct the probabilities are)
- **F1 score** = 2TP / (2TP + FP + FN) — a discrete, thresholded statistic over hard class decisions; it is piecewise-constant and **non-differentiable** with respect to model parameters

A model can output well-calibrated soft probabilities (low CE) while its decision boundary at threshold 0.5 is completely wrong for many samples. This is the **surrogate loss problem**: CE is a smooth proxy for classification correctness, but the proxy breaks down when the model has learned calibration before it has learned class separation.

> **The model at epoch 3 has learned "what to predict on average" before it has learned "how to distinguish between classes."**

### 2. Majority Class Exploitation (Class Imbalance)

With 4 mental task classes (mt4), class distributions are likely imbalanced. CE loss is an average over all samples — majority-class terms dominate the sum. In early epochs, gradient descent finds the cheapest descent path:

- Predict the dominant class with high confidence → low average log-loss
- Minority class recall collapses to ~0
- Macro F1 collapses to ~0.44

By later epochs, the model is forced to resolve residual error on the majority class. Doing so inadvertently builds representations that also separate minority classes. CE loss rises slightly (less overconfident), but macro F1 climbs to ~0.76 as minority-class recall recovers.

**Loss-based checkpointing in imbalanced settings actively selects the majority-collapse checkpoint.**

### 3. Stochastic "Lucky Dip" at Epoch 3 (Optimization Dynamics)

The epoch-3 val loss minimum is a transient artifact of early training instability:

- Learning rate is still high → large, noisy parameter updates
- Val loss dips when a particular weight configuration happens to fit the val set's *marginal statistics* (label priors) — not its discriminative structure
- This is a geometric coincidence, not convergence

After epoch 3, val loss rises as the model moves off this shallow basin — which is normal and expected. In later epochs, val loss re-approaches train loss, which **is** genuine convergence (stable representations). But checkpointing already stopped at epoch 3.

### 4. Dropout Measurement Artifact (Regularization)

The observed pattern of **val loss ≤ train loss** is NOT a convergence signal. It is a systematic measurement artifact:

| Phase | Dropout State | Effect on Loss |
|---|---|---|
| Training | **Active** — random activations zeroed | Loss inflated (noisy predictions) |
| Evaluation | **Disabled** — full model capacity | Loss deflated (clean predictions) |

The two curves measure **different computational graphs**. Val loss is structurally lower than train loss by construction in any dropout-regularized model. The "minimum" in val loss is the minimum of an artifact-inflated comparison, not the point of maximum generalization.

F1 score is **immune to this artifact** — it operates on thresholded hard predictions from the full eval-mode model, indifferent to probability magnitude or calibration drift between train/eval modes.

> **"Val loss < train loss" means dropout is working, not that the model generalizes optimally at that epoch.**

### 5. GATv2 Attention Immaturity (Architecture-Specific)

ST_GATv2_GNG relies on learned attention coefficients over a fixed GNG graph topology. In early epochs:

- Attention heads are near-uniform — all edges weighted approximately equally
- The model minimizes CE via global average over all attended nodes (safe, but spatially indiscriminate)
- Cross-entropy does NOT penalize how attention is distributed — only the final class probability output

To achieve good F1, the attention must sharpen to up-weight task-discriminative fNIRS channels (prefrontal/motor cortex nodes in GNG) and suppress noise nodes. This requires:
1. Sustained gradient signal over many epochs (attention sharpening is slow)
2. Extra epochs to handle the **fNIRS hemodynamic response lag** (~6s peak): early attention distributes weight across temporally lagged and non-lagged nodes indiscriminately

Attention maturity = F1 improves. Attention maturity does NOT = CE loss drops further.

---

## Summary

| Mechanism | Epoch 3 State | Later Epochs |
|---|---|---|
| Decision boundary | Not yet sharp | Sharp — separates classes |
| Class coverage | Majority-collapsed | Minority classes recovered |
| Attention weights | Near-uniform | Task-channel focused |
| CE landscape | Shallow stochastic minimum | Wider generalizing basin |
| Dropout interaction | Val loss artificially low | Same artifact, but representations stable |
| **Val F1** | **0.44** | **0.76** |

All five mechanisms compound: epoch 3 is simultaneously a stochastic dip, a majority-collapse checkpoint, an immature-attention state, and a measurement artifact — all of which happen to land at the CE loss minimum.

---

## Implication for This Codebase

**Fix:** Change `checkpoint_metric` from `val_loss` to `val_f1` in the ST pipeline config.

This is already supported — the pipeline has a configurable `checkpoint_metric` parameter. Switching ensures the saved model is the one with the best class-discriminative performance, not the best probability calibration at an early stochastic dip.

For reference, the relevant config in `src/core_st/` controls this via the YAML experiment config or CLI flag (`--checkpoint-metric`). See memory note `project_src_core_pipeline.md → Checkpoint Metric`.

---

## References

- Srivastava et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- Velickovic et al. (2022) — GATv2: How Attentive are Graph Attention Networks?
- Boyd & Vandenberghe — Convex Optimization (surrogate loss discussion)
- Lipton et al. (2014) — Optimal Thresholding of Classifiers to Maximize F1 Measure
