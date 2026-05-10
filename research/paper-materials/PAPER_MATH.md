# PAPER_MATH — Mathematical formulae for the Methods section

> Numbered formulas (E1..E10) used in §2.3.2–§2.3.3 of `PAPER_OUTLINE.md`. Each
> equation cross-references the source code line(s) that implement it, so any
> reviewer-driven change in the manuscript is reproducibly traceable to a
> commit. **Do not paraphrase.** When the SPEC plan needs to typeset an
> equation, lift the LaTeX block here verbatim.
>
> **Notation convention.** Bold lower-case = vectors (e.g. **x** ∈ ℝᵈ).
> Bold upper-case = matrices. Plain italic = scalars. Sub-scripts `i, j` index
> channels; `k` indexes time-windows; `t` indexes time-samples; `B` denotes a
> batch.
>
> **Cohort facts referenced.** N=23 prefrontal channels; T=326 timepoints
> (≈ 32 s @ 10.17 Hz); n_subj = 62 (HC=33, GAD=29).

---

## E1 — Per-channel statistical features (SG node features and ST window-stat module)

For each channel *i* and a 1-D signal segment **x**ᵢ ∈ ℝᴺ (full trial in SG, single window in ST), the six statistical features are

$$
\boxed{
\begin{aligned}
\mu_i      &= \frac{1}{N}\sum_{t=1}^{N} x_{i,t}                                    \\[2pt]
m_{i}^{\min} &= \min_{t}\, x_{i,t},\qquad m_{i}^{\max} = \max_{t}\, x_{i,t}        \\[2pt]
\sigma_i^2 &= \frac{1}{N}\sum_{t=1}^{N}\bigl(x_{i,t}-\mu_i\bigr)^{2}                \\[2pt]
g_{1,i}    &= \frac{m_{3,i}}{\sigma_i^{3}}, \qquad
g_{2,i}    \;=\; \frac{m_{4,i}}{\sigma_i^{4}}                                       \\[2pt]
\text{where } m_{k,i} &= \frac{1}{N}\sum_{t=1}^{N}\bigl(x_{i,t}-\mu_i\bigr)^{k}
\end{aligned}
}
$$

- $g_1$ is **Pearson skewness** (3rd standardised central moment), **not** Fisher-corrected.
- $g_2$ is **Pearson kurtosis** (4th standardised central moment), **not** excess kurtosis (no `–3`).
- Population variance is used (denominator *N*, not *N−1*).
- Numerical guard: when $\sigma_i^2 \le \varepsilon$ (with $\varepsilon=10^{-15}$ in SG, $10^{-8}$ in ST), $g_{1,i}, g_{2,i}$ are set to $0$ (ST: `nan_to_num`) or `NaN` (SG: replaced by the fold's training mean during `StandardizeGraphFeatures`).

**Source code:**
- SG: [`src/core/utils.py:31–59 compute_statistical_features`](../../src/core/utils.py).
- ST: [`src/core_st/models.py:103–119 _compute_window_stats`](../../src/core_st/models.py).

The two implementations are mathematically identical; ST uses `(centered**3).mean(-1) / (std**3)` with `std = sqrt(var).clamp(min=ε)`, which is algebraically `m₃ / σ³`.

---

## E2 — Pearson correlation matrix (edge coupling)

For the 1-D signals of all C=23 channels arranged as **X** ∈ ℝ^{C×N},

$$
\boxed{
\;C_{ij} \;=\; \mathrm{clip}\!\left(\frac{(\mathbf{x}_i-\mu_i)^{\!\top}(\mathbf{x}_j-\mu_j)}{\sqrt{\sum_{t}(x_{i,t}-\mu_i)^{2}}\;\sqrt{\sum_{t}(x_{j,t}-\mu_j)^{2}}},\; -1,\; +1\right)
\;}
$$

with $C_{ii}=1$ by construction. Computed on the **raw** trial (not z-scored), since Pearson is invariant to affine per-channel rescaling.

**Source:** [`src/core/utils.py:62–77 pearson_correlation_matrix`](../../src/core/utils.py); identical implementation in [`src/core_st/utils.py`](../../src/core_st/utils.py). Both SG and ST graphs use this same matrix.

---

## E3 — Welch magnitude-squared coherence matrix (edge coupling, frequency domain)

For each pair $(i, j)$ and frequency bin *f*,

$$
\boxed{
\;\mathrm{Coh}_{ij}(f) \;=\; \frac{\bigl|S_{ij}(f)\bigr|^{2}}{\,S_{ii}(f)\, S_{jj}(f)\,}\;\in\;[0,1]
\;}
$$

where $S_{ij}(f)$ is the Welch-averaged cross-spectral density estimated from L overlapping segments of length $L_{\text{seg}} = \lfloor N/3 \rfloor$ with 50 % overlap and a Hann window:

$$
S_{ij}(f) \;=\; \frac{1}{L}\sum_{\ell=1}^{L} X_{i,\ell}(f)\,\overline{X_{j,\ell}(f)},
\qquad
X_{i,\ell}(f) \;=\; \mathrm{rFFT}\!\bigl[w(t)\bigl(x_{i,t}^{(\ell)} - \overline{x_i^{(\ell)}}\bigr)\bigr]
$$

with $w(t) = \tfrac{1}{2}\bigl[1 - \cos(2\pi t/L_{\text{seg}})\bigr]$ (Hann).

The **scalar pairwise coherence** stored on each edge is the average over the non-DC, non-Nyquist bins:

$$
\mathrm{Coh}_{ij} \;=\; \frac{1}{F'-2}\sum_{f \in \{f_2, \ldots, f_{F'-1}\}} \mathrm{Coh}_{ij}(f), \qquad \mathrm{Coh}_{ii}=1.
$$

For T=326 samples, $L_{\text{seg}}=\lfloor 326/3\rfloor=108$, step 54, giving $L=5$ overlapping segments.

**Source:** [`src/core/utils.py:80–136 _hann_window, coherence_matrix`](../../src/core/utils.py). DC bin `0` and Nyquist bin `F-1` are excluded (line 132).

---

## E4 — Edge selection rule and edge feature

Given the matrices $\mathbf{C}$ (E2) and $\mathbf{Coh}$ (E3) and a threshold $\tau_{\text{corr}}=0.1$,

$$
\boxed{
\;(i,j)\in E \;\iff\; \bigl|C_{ij}\bigr| \;\ge\; \tau_{\text{corr}},
\qquad
\mathbf{e}_{ij} \;=\; \bigl(\mathrm{Coh}_{ij},\; |C_{ij}|\bigr)\;\in\;\mathbb{R}^{2}
\;}
$$

Self-loops are added for every node ($i=j$). Edges are stored as **directed**
(both $i\!\to\!j$ and $j\!\to\!i$ in `edge_index`), since the GATv2 attention
coefficient is asymmetric. The signed correlation **sign** is intentionally
discarded — it was tested in the CORAL ablation (project memory
`project_coral_results.md`) and reverted because it hurt 5-fold HBO F1 by ~9.7
pp.

**Source:** [`src/core/dataset.py:74–95 _build_graph`](../../src/core/dataset.py); same logic in [`src/core_st/dataset.py:74–95`](../../src/core_st/dataset.py).

---

## E5 — SG node feature vector (`FlexibleGATNet` input)

For each channel *i*, the SG model receives the static 6-dimensional descriptor

$$
\boxed{
\;\mathbf{x}_i^{\text{SG}} \;=\; \bigl(\mu_i,\, m_i^{\min},\, m_i^{\max},\, g_{1,i},\, g_{2,i},\, \sigma_i^{2}\bigr)\;\in\;\mathbb{R}^{6},
\qquad \mathbf{X}^{\text{SG}}\in\mathbb{R}^{23\times 6}.
\;}
$$

Computed once per trial on the **full** [23, 326] signal. Per-fold standardisation (E10) is applied to both **X** and edge_attr.

**Source:** [`src/core/dataset.py:53–60 _build_graph` — column ordering](../../src/core/dataset.py).

---

## E6 — ST input and window-stat transform (`WindowedSpatioTemporalGATNet`)

The ST model consumes the **raw** time series after **per-channel z-scoring per trial**:

$$
\tilde{x}_{i,t} \;=\; \frac{x_{i,t} - \mu_i}{\sigma_i+\varepsilon}, \qquad \mathbf{X}^{\text{ST}} = \tilde{\mathbf{X}}\in\mathbb{R}^{23\times 326}.
$$

Inside the model's `forward()`, the time axis is unfolded into K overlapping windows of size $W$ with stride $S$:

$$
K \;=\; \left\lfloor\frac{T-W}{S}\right\rfloor + 1
\quad\Longrightarrow\quad
\tilde{\mathbf{X}}\in\mathbb{R}^{23\times T}\;\xrightarrow{\;\mathrm{unfold}(W,S)\;}\;\mathbf{W}\in\mathbb{R}^{23\times K\times W}.
$$

Each window is reduced to a 6-feature vector via E1, applied along the window axis:

$$
\boxed{
\;\mathbf{Z} \;=\; \phi_{6}(\mathbf{W})\;\in\;\mathbb{R}^{23\times K\times 6},
\qquad
\phi_{6}(\mathbf{w}) = (\mu, m^{\min}, m^{\max}, \sigma^{2}, g_{1}, g_{2}).
\;}
$$

For the paper-headline configuration ($W=16$, $S=8$, $T=326$): $K=\lfloor(326-16)/8\rfloor+1=39$.

The fixed 6-dim output is what makes `in_channels=6` an architectural invariant rather than a hyperparameter.

**Source:** [`src/core_st/dataset.py:_build_graph`](../../src/core_st/dataset.py) (z-scoring); [`src/core_st/models.py:103–119 _compute_window_stats`](../../src/core_st/models.py); [`src/core_st/models.py:155–162 forward`](../../src/core_st/models.py) (unfold).

---

## E7 — GATv2 message-passing layer (Brody et al., 2022)

For each layer $\ell$, given node embeddings $\mathbf{h}_i^{(\ell)}\in\mathbb{R}^{d_\ell}$ and edge features $\mathbf{e}_{ij}\in\mathbb{R}^{d_e}$, the **GATv2** update with edge conditioning is

$$
\boxed{
\;\begin{aligned}
\boldsymbol{\eta}_{ij}^{(\ell)} &= \mathbf{W}_\ell^{l}\mathbf{h}_i^{(\ell)} + \mathbf{W}_\ell^{r}\mathbf{h}_j^{(\ell)} + \mathbf{W}_\ell^{e}\mathbf{e}_{ij}\\[2pt]
\alpha_{ij}^{(\ell)} &= \mathrm{softmax}_{j\in\mathcal{N}(i)\cup\{i\}}\Bigl(\mathbf{a}_\ell^{\top}\,\mathrm{LeakyReLU}\bigl(\boldsymbol{\eta}_{ij}^{(\ell)}\bigr)\Bigr)\\[2pt]
\tilde{\mathbf{h}}_i^{(\ell+1)} &= \Big\Vert_{m=1}^{H}\sum_{j\in\mathcal{N}(i)\cup\{i\}}\alpha_{ij}^{(\ell,m)}\,\mathbf{W}_\ell^{m,r}\mathbf{h}_j^{(\ell)} \quad (H \text{ heads, concatenated})\\[2pt]
\mathbf{h}_i^{(\ell+1)} &= \mathrm{Dropout}\Bigl(\mathrm{ELU}\bigl(\mathrm{Norm}\bigl(\tilde{\mathbf{h}}_i^{(\ell+1)}\bigr)\bigr)\Bigr) \;+\; \mathbf{P}_\ell\mathbf{h}_i^{(\ell)} \quad\text{(residual)}
\end{aligned}\;}
$$

Where $\mathbf{P}_\ell$ is a learned linear projection if $d_{\ell+1}\ne d_\ell$, else identity. **Concat=True** is used (multi-head outputs are concatenated, not averaged). LeakyReLU before the attention vector $\mathbf{a}_\ell$ is the GATv2 modification of the original GAT (Velickovic et al., 2018); cite **Brody, Alon & Yahav, ICLR 2022** for this layer.

**SG configuration** (per `experiments/spatial_graph/.../config.yaml`):
- $\ell=1$: $d_1=112$ filters per head, $H=8$ → $d_{\text{out},1}=896$. Residual proj `Linear(6, 896)`.
- $\ell=2$: $d_2=80$ filters per head, $H=6$ → $d_{\text{out},2}=480$. Residual proj `Linear(896, 480)`.
- `use_norm=False`, `dropout=0.3`, `edge_dim=2`.

**ST configuration** (paper-headline 20260509 sweep):
- Both layers identical: $d=80$, $H=2$ → $d_{\text{out}}=160$ per layer.
- `use_norm=True`, `norm_type=batch`, `use_residual=False`, `dropout=0.3`.
- Weights are **shared across all K=39 windows** (single GATv2 stack reused).

**Source:** PyTorch Geometric `GATv2Conv` (v2.7.0); wrapping logic in [`src/core/models.py:FlexibleGATNet`](../../src/core/models.py) and [`src/core_st/models.py:_spatial_encode`](../../src/core_st/models.py).

---

## E8 — ST temporal encoder: GRU + additive attention over windows

After per-window pooling and projection, the K=39 window embeddings $\{\mathbf{g}_k\}_{k=1}^{K}$ are processed by a single-layer GRU (Cho et al., 2014):

$$
\begin{aligned}
\mathbf{r}_k &= \sigma\bigl(\mathbf{W}_{r}\mathbf{g}_k + \mathbf{U}_{r}\mathbf{h}_{k-1}+\mathbf{b}_r\bigr)\\
\mathbf{z}_k &= \sigma\bigl(\mathbf{W}_{z}\mathbf{g}_k + \mathbf{U}_{z}\mathbf{h}_{k-1}+\mathbf{b}_z\bigr)\\
\tilde{\mathbf{h}}_k &= \tanh\bigl(\mathbf{W}_{h}\mathbf{g}_k + \mathbf{U}_{h}(\mathbf{r}_k\odot\mathbf{h}_{k-1})+\mathbf{b}_h\bigr)\\
\mathbf{h}_k &= (1-\mathbf{z}_k)\odot\mathbf{h}_{k-1} + \mathbf{z}_k\odot\tilde{\mathbf{h}}_k.
\end{aligned}
$$

A **Bahdanau-style additive attention** then weights each window's hidden state $\mathbf{h}_k$ to produce the trial-level context vector $\mathbf{c}\in\mathbb{R}^{d_h}$:

$$
\boxed{
\;\begin{aligned}
\mathbf{e}_k &= \tanh\bigl(\mathbf{W}_v\mathbf{h}_k\bigr)\\[2pt]
\alpha_k    &= \mathrm{softmax}_{k=1..K}\bigl(\mathbf{u}_v^{\top}\mathbf{e}_k\bigr)\\[2pt]
\mathbf{c}  &= \sum_{k=1}^{K}\alpha_k\,\mathbf{h}_k.
\end{aligned}\;}
$$

The temporal-attention weights $\boldsymbol{\alpha}\in\mathbb{R}^{K}$ are exposed by `model.explain()` for XAI in §3.3 and constitute the *primary* native attention path (cf. `docs/SPEC_xai_graph.md` §3.2).

For the paper-headline configuration: $d_h=192$, single-layer GRU, `temporal_layers=1`, `batch_first=True`.

**Source:** [`src/core_st/models.py:78–88, 178–195 forward`](../../src/core_st/models.py).

---

## E9 — Classifier head and loss

Both architectures share the same head:

$$
\hat{\mathbf{y}} \;=\; \mathrm{softmax}\!\bigl(\mathbf{W}_{\text{cls}}\,\mathrm{ELU}\!\bigl(\mathrm{Dropout}(\mathbf{W}_{\text{fc}}\mathbf{c})\bigr)\bigr)\;\in\;\Delta^{1}
$$

with two output logits $\hat{\mathbf{y}}=[\hat{p}_{\text{HC}},\hat{p}_{\text{GAD}}]$. The training objective is the **standard binary cross-entropy** computed over the two-logit softmax (PyTorch `CrossEntropyLoss`):

$$
\boxed{
\;\mathcal{L}_{\text{CE}} \;=\; -\frac{1}{B}\sum_{n=1}^{B}\sum_{c\in\{0,1\}} \mathbb{1}[y_n=c]\,\log \hat{p}_{n,c}.
\;}
$$

Class weights and `FocalLoss` (Lin et al., 2017) are implemented but **disabled** in the paper-headline runs (`use_focal_loss=False`, `use_class_weights=False`); class imbalance is mild (33/29).

**Source:** [`src/core_st/training.py FocalLoss + train_epoch`](../../src/core_st/training.py); [`src/core/training.py`](../../src/core/training.py).

---

## E10 — Per-fold leakage-free standardisation (the central reproducibility guarantee)

Let $\mathcal{D}_{\text{train}}^{(f)}$ denote the set of training graphs in fold $f$ (subject-level partitioning, see §2.3.5). The fold-specific feature mean and standard deviation are **derived from the training graphs only**:

$$
\boxed{
\;\begin{aligned}
\boldsymbol{\mu}^{(f)}_{x},\boldsymbol{\sigma}^{(f)}_{x} &= \mathrm{mean,std}\Bigl(\bigcup_{G\in\mathcal{D}_{\text{train}}^{(f)}}\mathbf{X}_G,\; \text{axis = 0}\Bigr)\\[2pt]
\boldsymbol{\mu}^{(f)}_{e},\boldsymbol{\sigma}^{(f)}_{e} &= \mathrm{mean,std}\Bigl(\bigcup_{G\in\mathcal{D}_{\text{train}}^{(f)}}\mathbf{e}_G,\; \text{axis = 0}\Bigr).
\end{aligned}\;}
$$

The same statistics are then applied — without recomputation — to **both** train and validation graphs in fold $f$:

$$
\mathbf{X}_G \;\leftarrow\; \frac{\mathbf{X}_G - \boldsymbol{\mu}^{(f)}_{x}}{\boldsymbol{\sigma}^{(f)}_{x}+\varepsilon},
\qquad
\mathbf{e}_G \;\leftarrow\; \frac{\mathbf{e}_G - \boldsymbol{\mu}^{(f)}_{e}}{\boldsymbol{\sigma}^{(f)}_{e}+\varepsilon},
\qquad \forall\, G\in\mathcal{D}_{\text{train}}^{(f)}\cup\mathcal{D}_{\text{val}}^{(f)}.
$$

**Critical SG vs ST difference:**
- **SG** standardises **both** $\mathbf{X}$ and edge_attr.
- **ST** standardises **only** edge_attr ($\mathbf{X}$ is already per-trial z-scored at dataset build time, see E6).

This contract — *training-set statistics computed once, applied once* — is the operationalisation of the "leak-free" claim made in §2.3.2.1.3 and §2.3.5 of the outline.

**Source:** [`src/core/dataset.py:139 compute_stats`](../../src/core/dataset.py); [`src/core_st/dataset.py:139 compute_stats`](../../src/core_st/dataset.py); [`src/core/transforms.py StandardizeGraphFeatures`](../../src/core/transforms.py); [`src/core_st/transforms.py StandardizeGraphFeatures`](../../src/core_st/transforms.py).

---

## Equation summary table (for SPEC plan cross-reference)

| ID | What it defines | Source code | Used in §  |
|---|---|---|---|
| E1 | Six per-channel statistical features (μ, min, max, σ², g₁, g₂) | `src/core/utils.py:31–59`, `src/core_st/models.py:103–119` | §2.3.2.1.1 (SG node features), §2.3.2.1.2 (ST window stats) |
| E2 | Pearson correlation matrix | `src/core/utils.py:62–77` | §2.3.2.1 (edge construction) |
| E3 | Welch magnitude-squared coherence | `src/core/utils.py:80–136` | §2.3.2.1 (edge construction) |
| E4 | Edge selection rule + 2-D edge feature | `src/core/dataset.py:74–95` | §2.3.2.1 |
| E5 | SG node feature vector | `src/core/dataset.py:53–60` | §2.3.2.1.1 |
| E6 | ST raw input + window-stat transform | `src/core_st/models.py:103–162` | §2.3.2.1.2 |
| E7 | GATv2 message passing (with edge cond., residual, multi-head) | PyG GATv2Conv + `models.py` | §2.3.3.1, §2.3.3.2 |
| E8 | GRU + additive attention over K windows | `src/core_st/models.py:78–88, 178–195` | §2.3.3.2 |
| E9 | Classifier head + cross-entropy loss | `src/core_st/training.py`, `src/core/training.py` | §2.3.3 |
| E10 | Per-fold leakage-free standardisation | `src/core/dataset.py:139`, `src/core_st/dataset.py:139` | §2.3.2.1.3, §2.3.5 |

---

## References to cite alongside these formulas

| Equation | Citation key (TBD in `refs.bib`) | Reason |
|---|---|---|
| E7 | `velickovic2018graph` | Original GAT paper |
| E7 | `brody2022how` | GATv2 (the layer actually used) |
| E8 | `cho2014learning` | GRU |
| E8 | `bahdanau2015neural` | Additive attention |
| E9 | `lin2017focal` | Focal loss (cited as alternative — not used in headline) |
| E10 | `varoquaux2017assessing` | Leakage in cross-validation (justifies the leak-free contract) |
| E2 | `pearson1895note` (or modern stats textbook) | Pearson correlation (rarely cited explicitly in ML papers) |
| E3 | `welch1967use` | Welch's method for spectral estimation |
| E3 | `carter1987coherence` | Magnitude-squared coherence interpretation |

These keys are placeholders; populate during P1.3 (Literature review + bibliography population).

---

## What the SPEC plan should do with this file

1. **Lift each equation block verbatim** (the `\boxed{…}` LaTeX) into the Methods section. Do not reformat.
2. **Number them E1..E10 in the manuscript** (matching this file).
3. **Reuse the source-code line references** as supplementary material so reviewers can audit every formula against the running code.
4. **Add the citations** in the table above to `refs.bib` during P1.3, using `citation-management` to verify DOI/year/journal.
