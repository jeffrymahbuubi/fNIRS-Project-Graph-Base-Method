# fNIRS Dataset Quality Report

**Generated:** 2026-04-30  
**Assessed by:** Automated quality scan + Homer3 best-practice cross-reference  
**Datasets:** `data/raw` (64 subjects, all 4 tasks) and `data/additional-raw` (11 subjects, GNG+SS)  
**Basis:** Signal quality metrics computed from HbO/HbR CSV files + raw .nirs intensity CV (additional-raw only)

---

## 1. Quality Metrics Used

| Metric | Source | Threshold | Meaning |
|--------|--------|-----------|---------|
| **Mean spike count** | HbO CSV (amplitude jumps > 5× mean diff) | >3000 severe, 1500–3000 moderate | Motion artifact density |
| **HbO/HbR correlation** | Channel-wise Pearson r | >0.5 severe, >0.2 moderate | Systemic noise dominance (ideal: negative) |
| **Raw intensity CV** | .nirs file (additional-raw only) | >15% bad, 7.5–15% acceptable | Scalp contact / optode coupling quality |
| **Signal scale** | Mean \|HbO\| | 1–4×10⁻⁷ M ideal | Calibration / Beer-Lambert validity |

### Risk Classification

| Flag | Condition |
|------|-----------|
| **HIGH-RISK** | Score ≥ 4 (severe motion + systemic noise, or bad raw channels) |
| **WATCH** | Score 2–3 (moderate issues — usable but may degrade model performance) |
| **OK** | Score 0–1 (no significant issues detected) |

> **Important context:** The positive HbO/HbR correlation seen in many subjects is partially expected when no motion correction is applied (see §4). It does NOT automatically mean the data is unusable — but it does indicate that systemic physiological noise (heartbeat, Mayer waves) is not suppressed.

---

## 2. HIGH-RISK Subjects (11 total)

These subjects show a combination of high motion artifact density and/or strong systemic noise dominance. Their inclusion in training may introduce label-irrelevant signal variation.

### data/raw — Anxiety (2 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Primary Issue |
|---------|-------------|--------------|---------------|
| **AA011** | 4,493 | +0.479 | Severe motion + strong systemic noise |
| **EA055** | 3,438 | +0.441 | Severe motion + strong systemic noise |
| **LA063** | 3,879 | +0.604 | Severe motion + very strong systemic noise (highest corr in anxiety group) |

### data/raw — Healthy (4 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Primary Issue |
|---------|-------------|--------------|---------------|
| **AH024** | 3,443 | +0.560 | Severe motion + very strong systemic noise |
| **AH045** | 3,399 | +0.458 | Severe motion + strong systemic noise |
| **AH046** | 3,420 | +0.489 | Severe motion + strong systemic noise |
| **AH049** | 3,469 | +0.468 | Highest spike count in healthy group |

### data/additional-raw — Anxiety (4 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Raw CV (nirs) | Bad Ch | Primary Issue |
|---------|-------------|--------------|---------------|--------|---------------|
| **AA093** | 4,867 | +0.319 | 4.9% | 0 | Severe motion + moderate systemic noise |
| **AA094** | 3,209 | +0.321 | 6.8% | 1 | Severe motion + moderate systemic noise |
| **AA099** | 3,395 | −0.120 | 7.5% | 1 | Severe motion + 1 marginal channel (note: HbO/HbR corr is actually good) |
| **LA096** | 3,587 | +0.182 | 6.9% | 1 | Severe motion + 1 marginal channel |

> **Note on AA099:** Despite being HIGH-RISK due to motion artifacts and 1 marginal channel, the HbO/HbR correlation is negative (−0.12) which is the correct physiological direction. AA099's risk is primarily motion-artifact driven, not systemic noise. It may still be usable after motion correction.

---

## 3. WATCH Subjects (22 total)

Moderate concerns. Usable for training but should be monitored for outlier behavior during cross-validation.

### data/raw — Anxiety (8 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Watch Reason |
|---------|-------------|--------------|-------------|
| AA003 | 1,959 | +0.421 | Moderate motion + moderate noise |
| AA004 | 1,736 | +0.214 | Moderate motion + moderate noise |
| EA060 | 1,623 | +0.221 | Moderate motion + moderate noise |
| EA062 | 2,414 | +0.404 | Moderate-high motion + moderate noise |
| LA042 | 1,703 | +0.264 | Moderate motion + moderate noise |
| LA054 | 2,047 | +0.386 | Moderate motion + moderate noise |
| LA057 | 1,946 | +0.340 | Moderate motion + moderate noise |
| LA059 | 2,227 | +0.401 | Moderate motion + moderate noise |

### data/raw — Healthy (9 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Watch Reason |
|---------|-------------|--------------|-------------|
| AH015 | 1,816 | +0.256 | Moderate motion + noise |
| AH022 | 1,634 | +0.263 | Moderate motion + noise |
| AH025 | 2,612 | +0.478 | Moderate-high motion + high noise |
| AH027 | 1,904 | +0.344 | Moderate motion + moderate noise |
| AH028 | 2,204 | +0.374 | Moderate motion + moderate noise |
| AH030 | 2,028 | +0.277 | Moderate motion + moderate noise |
| AH032 | 2,462 | +0.408 | Moderate motion + moderate noise |
| AH035 | 2,501 | +0.324 | Moderate motion + moderate noise |
| AH040 | 2,162 | +0.620 | Highest noise in healthy WATCH group |

### data/additional-raw — Anxiety (5 subjects)

| Subject | Mean Spikes | HbO/HbR Corr | Raw CV | Watch Reason |
|---------|-------------|--------------|--------|-------------|
| AA090 | 5,870 | +0.124 | 6.5% | Very high motion (highest in add-raw) |
| AA092 | 3,889 | +0.082 | 4.7% | High motion |
| AA097 | 2,786 | +0.357 | 2.3% | Moderate motion + moderate noise |
| LA091 | 3,701 | +0.150 | 3.7% | High motion |
| LA095 | 2,583 | +0.406 | 9.3% | Moderate motion + moderate noise + worst raw CV in add-raw group |

---

## 4. OK Subjects (42 total)

No significant quality concerns detected.

**data/raw — Anxiety (OK):**  
AA002, AA005, AA006, AA007, AA008, AA013, AA041, AA056, AA064, EA012, EA016, EA061, LA051, LA052, LA053, LA058

**data/raw — Healthy (OK):**  
AH009, AH010, AH014, AH017, AH018, AH019, AH020, AH021, AH023, AH026, AH029, AH031, AH033, AH034, AH036, AH037, AH038, AH039, AH043, AH044, AH047, AH048, AH050

**data/additional-raw — Anxiety (OK):**  
AA089, AA098

---

## 5. Dataset-Wide Summary Statistics

### data/raw (64 subjects, GNG–1backWM–SS–VF tasks)

| Metric | Healthy (36 subj) | Anxiety (28 subj) | All |
|--------|-------------------|-------------------|-----|
| Mean spike count | ~1,870 | ~1,780 | ~1,830 |
| Mean HbO/HbR corr | +0.22 | +0.21 | +0.21 |
| Mean signal scale | 2.69×10⁻⁷ M | 2.03×10⁻⁷ M | 2.44×10⁻⁷ M |
| HIGH-RISK | 4 (11%) | 3 (11%) | 7 (11%) |
| WATCH | 9 (25%) | 8 (29%) | 17 (26%) |
| OK | 23 (64%) | 17 (61%) | 40 (63%) |

### data/additional-raw (11 subjects, GNG+SS tasks)

| Metric | Value |
|--------|-------|
| Mean raw intensity CV | 5.9% (good: <7.5%) |
| Mean spike count | ~3,600 (higher than data/raw) |
| Mean HbO/HbR corr | +0.24 |
| Mean signal scale | 2.02×10⁻⁷ M |
| HIGH-RISK | 4 (36%) |
| WATCH | 5 (45%) |
| OK | 2 (18%) |

> The additional-raw group has higher motion artifact counts than data/raw overall. This may reflect the more recent recording protocol or the combined-session design (longer continuous recording = more opportunity for motion).

---

## 6. Recommendations for AI Training

### High priority actions

1. **Exclude or closely monitor HIGH-RISK subjects** in your cross-validation splits.  
   Consider running your experiments both with and without these 11 subjects to quantify their impact on model performance.

2. **Note: HIGH-RISK ≠ automatic exclusion.**  
   After motion correction (Wavelet + CBSI), some HIGH-RISK subjects may improve to an acceptable quality level. Re-assess post-correction.

3. **Balance check across groups:**  
   HIGH-RISK distribution is roughly equal between groups (11% anxiety, 11% healthy in data/raw), so excluding them should not create significant class imbalance.

4. **data/additional-raw subjects (AA089–AA099) have not had motion correction applied.**  
   Their higher spike counts are expected. Process with Wavelet + CBSI before final inclusion.

### Specific subject recommendations

| Subject | Dataset | Recommendation |
|---------|---------|----------------|
| LA063 | raw/anxiety | Highest combined risk score — consider excluding |
| AH024 | raw/healthy | Highest noise in healthy group — consider excluding |
| AH049 | raw/healthy | Highest spike count in healthy group — consider excluding |
| AA089 | add-raw | Cleanest additional-raw subject — reliable training sample |
| AA098 | add-raw | Second cleanest — reliable training sample |
| AA090 | add-raw | Very high motion — prioritize for motion correction |

---

## 7. Processing Pipeline Gap (Root Cause)

The current pipeline (`nirs2csv_homer3.m`) applies:
```
Intensity → OD → Bandpass(0.01–0.5 Hz) → Conc(HbO/HbR/HbT)
```

It is **missing**:
- Channel quality check before conversion
- Motion artifact correction (Wavelet, CBSI, Spline)

The positive HbO/HbR correlation observed across most subjects is a direct consequence of this omission — physiological noise (cardiac, Mayer waves) at frequencies below 0.5 Hz passes through the bandpass filter and remains in the signal.

**Recommended addition** (no channel removal — see §8):
```
Intensity → OD → WaveletCorrect → CBSI → Bandpass(0.01–0.5 Hz) → Conc
```

---

## 8. Channel Preservation Policy

All 23 channels are **mandatory** for this study's 5×7 grid encoding. The `hmrIntensityPrune` function (which removes bad channels) is therefore **not applicable** here.

Instead:
- **Identify** bad channels from raw CV metrics (flagged in additional-raw above) for reporting purposes only.
- **Apply motion correction** (Wavelet + CBSI) on all 23 channels without removal.
- Bad channels flagged in raw CV (AA094 ch, AA099 ch, LA096 ch) will retain noise, but RBF interpolation in `generate_matrix()` provides partial spatial smoothing from neighboring channels.

---

*Report generated from automated quality scan. All metrics are based on GNG task (data/raw) and GNG+SS tasks (data/additional-raw). Full per-task, per-channel breakdown available in `/tmp/fnirs_quality_full.csv`.*

---

## 9. Pipeline Comparison: Old vs Motion-Corrected (GNG Task)

**Analysis date:** 2026-04-30  
**Scope:** GNG task, HbO signal, 62 shared subjects  
**Old pipeline source:** `data/processed-new/GNG` — Homer3 bandpass only (`nirs2csv_homer3.m`)  
**New pipeline source:** `data/processed-old-new-mc/GNG` — Wavelet + CBSI (`nirs2csv_homer3_mc.m`)  
**Processing settings applied to both:** `--apply-baseline --apply-zscore --save-preprocessed`

---

### 9.1 Pipeline Architecture Difference

| Stage | Old pipeline | New (MC) pipeline |
|-------|-------------|-------------------|
| 1. Intensity → OD | ✓ `hmrR_Intensity2OD` | ✓ `hmrR_Intensity2OD` |
| 2. Motion correction | ✗ absent | ✓ `hmrR_MotionCorrectWavelet` (IQR=1.5) |
| 3. Physiological noise | ✗ absent | ✓ `hmrR_CBSI` |
| 4. Bandpass 0.01–0.5 Hz | ✓ | ✓ |
| 5. Conc (Beer-Lambert) | ✓ `hmrR_OD2Conc` | ✓ `hmrR_OD2Conc` |

The old pipeline passes raw intensity directly to bandpass and Beer-Lambert conversion. Any motion artifact that falls within the 0.01–0.5 Hz passband (which most head-movement events do, as their duration overlaps the low-frequency task band) is **indistinguishable from neural hemodynamic signal** at the OD conversion stage.

**Wavelet MC** decomposes each OD channel into frequency sub-bands using a discrete wavelet transform and identifies artifact-contaminated coefficients as those exceeding `IQR × threshold` (1.5 in this run). Contaminated coefficients are suppressed before reconstruction. This targets the **amplitude and shape** of motion transients rather than their frequency content, making it effective even for motion events whose bandwidth overlaps the task-frequency range.

**CBSI (Correlation-Based Signal Improvement)** exploits the physiological constraint that HbO and HbR must be negatively correlated in genuine neurovascular responses. It decomposes the OD signal into a component parallel to motion (positive HbO–HbR covariation) and a component orthogonal to motion (negative HbO–HbR covariation), then retains only the latter. By construction, CBSI forces r(HbO, HbR) = −1.0 in the output.

---

### 9.2 Quantitative Signal Quality Metrics

All metrics computed on z-score-normalised HbO epochs (channel-wise z-score applied in `processor_cli.py`).

| Metric | Old pipeline | MC pipeline | Δ | Direction |
|--------|-------------|-------------|---|-----------|
| Temporal std (mean ± std) | 0.9252 ± 0.0268 | 0.8945 ± 0.0306 | −3.3% | Lower = less noise |
| Temporal std (median) | 0.9305 | 0.8967 | −3.6% | |
| Temporal std (95th pct) | 0.9603 | 0.9386 | −2.3% | |
| Temporal std (maximum) | 0.9705 | 0.9494 | −2.2% | |
| Spike ratio (>3σ events) | 0.0014 ± 0.0012 | 0.0011 ± 0.0011 | **−23.4%** | Lower = fewer spikes |
| Spike ratio — healthy only | 0.0015 | 0.0013 | −16% | |
| Spike ratio — anxiety only | 0.0013 | 0.0009 | −31% | |
| HbO–HbR r (mean ± std) | −0.025 ± 0.613 | **−1.000 ± 0.000** | — | Negative = physiological |
| Extreme epochs (\|peak\| > 10σ) | 0 | 0 | 0 | — |
| Subjects with MC regression | — | 2 / 62 | ΔΔ < 0.002 | Negligible |

**Notes on interpretation:**

- **Temporal std** measures residual variability after z-scoring. Since each channel is individually normalised to unit variance, a mean std < 1.0 indicates that cropping the preparation window (3 s for GNG) reduced apparent variance — this is expected. The MC pipeline reduces this further because Wavelet MC has already suppressed high-variance transient events before epoching.

- **Spike ratio** is the fraction of time-samples exceeding 3 standard deviations from the channel mean within each epoch. This is the clearest per-epoch indicator of residual motion contamination. A 23% overall reduction (31% in the anxiety group) is practically meaningful given the class imbalance correction requirements of this study.

- **HbO–HbR correlation** is the most diagnostically important metric. In the old pipeline the mean correlation is −0.025 with standard deviation 0.61, meaning many individual channel-epochs have **positive** HbO–HbR correlation. Positive correlation is unambiguously artifactual: it cannot arise from genuine neurovascular coupling and indicates that motion or systemic noise is the dominant signal component. In the MC pipeline, CBSI forces r = −1.0 by construction; see §9.4 for a discussion of this constraint.

---

### 9.3 Dataset Coverage Changes

| Group | Old pipeline subjects | MC pipeline subjects | Newly recovered |
|-------|-----------------------|---------------------|-----------------|
| Healthy (GNG) | 33 | 33 | — |
| Anxiety (GNG) | 29 | **31** | EA012, EA016 |
| **Total** | **62** | **64** | **+2 anxiety** |
| Total GNG epochs | 248 | 256 | +8 |

EA012 and EA016 are anxiety subjects that produced valid GNG epochs under the MC pipeline but failed to generate output in the old pipeline (likely due to missing or unmatched CSV files in the earlier batch). Both subjects are rated **OK** in the pre-MC quality assessment (§4), making them clean additions.

Subjects with zero GNG epochs in **both** pipelines (AH009, AH010, AH032, AA001–AA008) are early-protocol subjects recorded under a different task design; they are excluded from GNG analysis regardless of pipeline.

---

### 9.4 Subject-Level Improvement Profile

All ten highest-artifact subjects (by temporal std in the old pipeline) improved under MC:

| Subject | Group | Old std | MC std | Reduction |
|---------|-------|---------|--------|-----------|
| LA095 | anxiety | 0.9705 | 0.9494 | −2.2% |
| AH044 | healthy | 0.9645 | 0.9404 | −2.5% |
| AA064 | anxiety | 0.9610 | 0.9308 | −3.1% |
| AH047 | healthy | 0.9604 | 0.9484 | −1.2% |
| **AA092** | anxiety | 0.9578 | **0.8503** | **−11.2%** |
| AH036 | healthy | 0.9568 | 0.9230 | −3.5% |
| **AA093** | anxiety | 0.9552 | **0.8697** | **−9.0%** |
| **AA094** | anxiety | 0.9551 | **0.8973** | **−6.1%** |
| AH022 | healthy | 0.9540 | 0.9283 | −2.7% |
| LA096 | anxiety | 0.9533 | 0.9317 | −2.3% |

The three subjects with the **largest improvements** (AA092, AA093, AA094) are all from `data/additional-raw` — a cohort independently assessed as having the highest motion artifact burden in the pre-correction quality report (§5). This validates the Wavelet MC step specifically for the subjects it was most needed for.

Only 2 subjects (AH014, AH045) showed a marginal regression under MC (+0.0015 and +0.0008 std respectively), well below any meaningful threshold. No subject was degraded enough to warrant exclusion from the MC dataset.

---

### 9.5 CBSI Constraint: Limitation and Mitigation

CBSI enforcing r(HbO, HbR) = −1.0 is both its strength and its limitation:

**Why this is acceptable for the present study:**
- The classification target (anxiety vs healthy) is based on the **pattern and magnitude** of the hemodynamic response, not on the precise coupling ratio between HbO and HbR independently.
- HbT = HbO + HbR is a linear combination that is **not** constrained to zero by CBSI. HbT retains genuine hemodynamic amplitude information while the Wavelet+CBSI steps have already cleaned motion artifacts from both channels.
- Using HbT for the classification model avoids the CBSI artefact entirely while still benefiting from the motion-corrected signal.

**Recommended data type for downstream modelling:** `hbt` from `processed-old-new-mc`. This provides motion-corrected, physiologically valid total hemoglobin without the r = −1 constraint on individual channels.

---

### 9.6 Pipeline Recommendation

| Decision | Recommendation |
|----------|---------------|
| **Primary dataset** | `data/processed-old-new-mc` |
| **Primary signal** | `hbt` (HbO + HbR, avoids CBSI r=−1 constraint) |
| **Fallback signal** | `hbo` (if HbT analysis is unavailable) |
| **Exclude old pipeline** | Yes — `data/processed-new` should not be used for new experiments |
| **Subjects to monitor** | See HIGH-RISK list in §2; all improved under MC but remain the noisiest subjects |
| **Subjects to exclude** | LA063 (highest combined risk), AH024 (highest noise in healthy) — optional, see §6 |

**Summary justification:**
1. The old pipeline produces near-zero HbO–HbR correlation (r = −0.025 ± 0.61), revealing that motion and systemic physiological noise — not neural hemodynamics — is the dominant signal component for a significant fraction of subjects and epochs.
2. The MC pipeline reduces spike occurrence by 23% overall (31% in the anxiety group), which is the group where artifact contamination most risks corrupting the classification target.
3. The MC pipeline recovers 2 additional anxiety subjects (EA012, EA016) and produces the largest quality gains in the highest-burden subjects from `data/additional-raw`.
4. No subject is degraded to an unacceptable level by the MC pipeline.
5. The 64-subject MC GNG dataset provides a 3.2% larger training pool than the 62-subject old-pipeline dataset, with measurably cleaner signal throughout.
