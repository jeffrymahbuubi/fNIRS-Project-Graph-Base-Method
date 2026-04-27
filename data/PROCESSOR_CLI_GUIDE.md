# fNIRS Data Processor - Command Line Interface

This document describes how to use the `processor_cli.py` script with command-line arguments for processing fNIRS RAW data files.

## Features

The processor supports:
- **Single subject processing**: Process one subject at a time
- **Batch processing**: Process multiple subjects from a JSON configuration
- **Data validation**: Visualize processed data for quality control
- **5×7 grid mapping**: Convert 23-channel fNIRS signals to spatial grid tensors `(T, 5, 7)` with Gaussian RBF interpolation for ViT-based training
- **Flexible configuration**: Control all processing parameters via command-line arguments

## Installation

Ensure you have the required dependencies:
```bash
pip install mne pandas numpy matplotlib
```

## Usage Modes

### 1. Single Subject Processing

Process a single subject's data:

```bash
python processor_cli.py \
  --mode single \
  --root-dir /path/to/raw_data \
  --subject AH001 \
  --group healthy \
  --task GNG \
  --data-type hbo \
  --apply-baseline \
  --apply-zscore
```

**Required arguments for single mode:**
- `--mode single`
- `--root-dir`: Path to directory containing raw data
- `--subject`: Subject ID
- `--group`: Either `healthy` or `anxiety`

**Optional arguments:**
- `--output-dir`: Save metadata to this directory
- `--task`: Task type (default: `GNG`, choices: `GNG`, `1backWM`, `VF`, `SS`)
- `--data-type`: Data type (default: `hbo`, choices: `hbo`, `hbr`, `hbt`)
- `--apply-baseline`: Apply baseline correction
- `--apply-zscore`: Apply z-score normalization
- `--save-preprocessed`: Save epoch data
- `--save-format`: Format for saving preprocessed data (default: `npy`, choices: `npy`, `txt`)
- `--montage-file`: Path to custom montage file
- `--ppf`: Partial pathlength factor (default: 6.0)

### 2. Batch Processing

Process multiple subjects from a JSON configuration file:

```bash
python processor_cli.py \
  --mode batch \
  --root-dir /path/to/raw_data \
  --output-dir /path/to/processed_data \
  --subjects-json subjects.json \
  --task GNG \
  --data-type hbo \
  --apply-baseline \
  --apply-zscore \
  --save-preprocessed \
  --save-format txt
```

**Required arguments for batch mode:**
- `--mode batch`
- `--root-dir`: Path to directory containing raw data
- `--output-dir`: Path to save processed data
- `--subjects-json`: Path to JSON file with subject configuration

**JSON file format** (`subjects.json`):
```json
{
  "healthy": ["AH001", "AH002", "AH003"],
  "anxiety": ["AA001", "AA002", "AA003"]
}
```

### 3. Data Validation

Visualize processed data for a specific subject:

```bash
python processor_cli.py \
  --mode validate \
  --output-dir /path/to/processed_data \
  --subject AH001
```

**Required arguments for validate mode:**
- `--mode validate`
- `--output-dir`: Path to processed data directory
- `--subject`: Subject ID to validate

This will display a plot showing the average HbO, HbR, and HbT signals across all epochs.

## Command-Line Arguments Reference

### Mode Selection
- `--mode {single,batch,validate}` **(required)**: Processing mode

### Directory Paths
- `--root-dir PATH`: Root directory containing raw data folders
- `--output-dir PATH`: Output directory for processed data

### Subject Selection
- `--subject ID`: Subject identifier (e.g., AH001)
- `--group {healthy,anxiety}`: Subject group
- `--subjects-json FILE`: JSON file with subject dictionary (for batch mode)

### Processing Parameters
- `--task {GNG,1backWM,VF,SS}`: Task type (default: GNG)
- `--data-type {hbo,hbr,hbt}`: Data type to process (default: hbo)
- `--apply-baseline`: Apply baseline correction (flag)
- `--apply-zscore`: Apply z-score normalization (flag)
- `--save-preprocessed`: Save preprocessed epochs (flag)
- `--save-format {npy,txt}`: Format for saving preprocessed data (default: npy)
- `--montage-file FILE`: Path to custom montage file
- `--ppf FLOAT`: Partial pathlength factor for Beer-Lambert law (default: 6.0)

### Grid Mapping (for ViT Training)
- `--use-grid-mapping`: Convert 23-channel data to 5×7 spatial grid using Gaussian RBF interpolation. Saves epochs as `(T, H, W)` instead of `(23, T)`. **Required for ViT-based training.**
- `--grid-size H,W`: Grid dimensions (default: `5,7`). Only used with `--use-grid-mapping`.
- `--no-interpolation`: Disable Gaussian RBF interpolation for empty grid cells (only with `--use-grid-mapping`).

### Plot Output
- `--save-plots`: Save `{task}_time_marker.png` and `{task}_evoked.png` per subject to `output-dir/{task}/{group}/{subject}/`.

### Logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging verbosity (default: INFO)

## Examples

### Example 1: Quick single subject processing
```bash
python processor_cli.py --mode single --root-dir ./data/raw_data --subject AH001 --group healthy
```

### Example 2: Batch processing with all preprocessing steps
```bash
python processor_cli.py \
  --mode batch \
  --root-dir ./data/raw_data \
  --output-dir ./data/processed_data \
  --subjects-json subjects.json \
  --task GNG \
  --data-type hbo \
  --apply-baseline \
  --apply-zscore \
  --save-preprocessed \
  --save-format txt \
  --log-level DEBUG
```

### Example 3: Process HbT (total hemoglobin) data
```bash
python processor_cli.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --data-type hbt \
  --apply-baseline
```

### Example 4: Validate processed data
```bash
python processor_cli.py --mode validate --output-dir ./data/processed_data --subject AH001
```

### Example 5: Custom montage and PPF
```bash
python processor_cli.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --montage-file ./misc/brainproducts-RNP-BA-128-custom.elc \
  --ppf 7.5
```

### Example 6: Save preprocessed data as text files
```bash
python processor_cli.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --save-preprocessed \
  --save-format txt
```

### Example 7: Batch processing with 5×7 grid mapping for ViT training
Generates `(T, 5, 7)` `.npy` files ready for `src/core/main.py`:
```bash
python data/processor_cli.py \
  --mode batch \
  --root-dir ./data/raw \
  --output-dir ./data/processed \
  --subjects-json data/subjects.json \
  --task GNG \
  --data-type hbt \
  --save-preprocessed \
  --apply-zscore \
  --use-grid-mapping \
  --grid-size 5,7
```

> **Note**: Without `--use-grid-mapping`, epochs are saved as raw `(23, T)` arrays. With it, they are saved as `(T, H, W)` grid tensors. The ViT training pipeline (`src/core/main.py`) requires grid-mapped data.

### Example 8: Complete Batch Processing — All Tasks × All Data Types

Copy-paste any command below. All use `--apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping` as a standard preprocessing pipeline.

#### GNG

```bash
# GNG — HbO
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task GNG --data-type hbo \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# GNG — HbR
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task GNG --data-type hbr \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# GNG — HbT
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task GNG --data-type hbt \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping
```

#### 1backWM

```bash
# 1backWM — HbO
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task 1backWM --data-type hbo \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# 1backWM — HbR
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task 1backWM --data-type hbr \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# 1backWM — HbT
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task 1backWM --data-type hbt \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping
```

#### VF

```bash
# VF — HbO
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task VF --data-type hbo \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# VF — HbR
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task VF --data-type hbr \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# VF — HbT
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task VF --data-type hbt \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping
```

#### SS

```bash
# SS — HbO
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task SS --data-type hbo \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# SS — HbR
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task SS --data-type hbr \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping

# SS — HbT
python processor_cli.py --mode batch --root-dir raw/ --output-dir ./processed \
  --subjects-json ./subjects.json --task SS --data-type hbt \
  --apply-baseline --apply-zscore --save-preprocessed --use-grid-mapping
```

> **Tip**: Append `--save-plots` to any command above to also generate `*_time_marker.png` and `*_evoked.png` per subject alongside the processed epochs.

## Directory Structure

Expected raw data structure:
```
root_dir/
├── healthy/
│   ├── AH001/
│   │   └── GNG/
│   │       ├── *.nirs files
│   │       ├── *HbO.csv
│   │       └── *HbR.csv
│   └── AH002/
└── anxiety/
    └── AA001/
```

Output structure (batch mode with `--save-preprocessed`):
```
output_dir/
└── GNG/
    ├── healthy/
    │   └── AH001/
    │       ├── AH001.data          ← subject metadata (sfreq)
    │       └── hbt/
    │           ├── 0.npy           ← shape (23, T) without --use-grid-mapping
    │           ├── 1.npy           ← shape (T, 5, 7) with --use-grid-mapping
    │           └── ...
    └── anxiety/
        └── AA001/
```

> Epochs with `--use-grid-mapping` are directly consumable by `src/core/datasets.py` and the ViT training pipeline.

## Compress & Transfer Processed Data

### Compress

```bash
tar -czf processed.tar.gz -C /path/to/data processed
```

### Extract

```bash
tar -xzf processed.tar.gz
# or to a specific location:
tar -xzf processed.tar.gz -C /target/directory
```

> **Tip**: Use `-cjf` / `-xjf` for `.tar.bz2` (smaller) or `-cJf` / `-xJf` for `.tar.xz` (smallest but slowest).

## Excluded Subjects

Sixteen subjects present in `data/raw/` were excluded from `subjects.json` and batch processing because they completed only a **partial set of cognitive tasks** (fewer than the required 4: `1backWM`, `GNG`, `SS`, `VF`). Including them would create an incomplete cross-task dataset and cause missing-file errors during batch processing.

### Group 1 — Early Protocol (10 subjects)

AA001–AA008 and their matched healthy controls AH009–AH010 were recruited under an early study protocol that used `AP` (anxiety-provoking) and `RP` (relaxation-provoking) tasks instead of `SS` and `GNG`. They have only `1backWM` + `VF` and are missing both `SS` and `GNG`.

| Subject | Group | Tasks Completed | Missing Task(s) |
|---------|-------|-----------------|-----------------|
| AA001 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA002 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA003 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA004 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA005 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA006 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA007 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AA008 | anxiety | 1backWM, VF | `SS`, `GNG` |
| AH009 | healthy | 1backWM, VF | `SS`, `GNG` |
| AH010 | healthy | 1backWM, VF | `SS`, `GNG` |

### Group 2 — Standard Protocol, Incomplete Task Set (6 subjects)

Subjects recruited under the standard protocol but with one or more tasks missing from their recording session.

| Subject | Group | Tasks Completed | Missing Task(s) |
|---------|-------|-----------------|-----------------|
| AA011 | anxiety | 1backWM, GNG, SS | `VF` |
| EA012 | anxiety | 1backWM, GNG, VF | `SS` |
| EA016 | anxiety | GNG, VF | `1backWM`, `SS` |
| LA053 | anxiety | 1backWM, GNG, SS | `VF` |
| AH032 | healthy | 1backWM, SS, VF | `GNG` |
| AH047 | healthy | GNG, SS, VF | `1backWM` |

All excluded subjects' raw data remains in `data/raw/` and can be used for single-task analyses if needed. The final dataset used for model training is **32 healthy + 16 anxiety = 48 subjects**, all with complete 4-task recordings.

## Tips

1. **Start with single mode** to test parameters on one subject before batch processing
2. **Use --log-level DEBUG** for detailed troubleshooting
3. **Validate your data** after processing to ensure quality
4. **Create a subjects.json** template for reproducible batch processing
5. **Choose save format wisely**: Use `npy` for faster loading in Python/NumPy, or `txt` for human-readable format and compatibility with other tools
6. **Save preprocessed data** only when needed to save disk space
7. **Use `--use-grid-mapping`** when preparing data for ViT training via `src/core/main.py` — without it, training will fail with a shape mismatch error
8. **Use `--data-type hbt`** for best classification performance (HbT outperforms HbO/HbR by 15–25pp per paper results)

## Programmatic Usage

The classes can be imported and used in Python scripts:

```python
from processor_cli import FNIRSDataProcessor, FNIRSDataset

# Single subject — raw (23, T) output
processor = FNIRSDataProcessor(
    root_dir='./data/raw',
    subject='AH001',
    group='healthy',
    task_type='GNG',
    data_type='hbt',
    apply_zscore=True,
    save_preprocessed=True,
)
epochs = processor.process()  # list of (23, T) arrays

# Single subject — grid-mapped (T, 5, 7) output for ViT training
processor = FNIRSDataProcessor(
    root_dir='./data/raw',
    subject='AH001',
    group='healthy',
    task_type='GNG',
    data_type='hbt',
    apply_zscore=True,
    save_preprocessed=True,
    use_grid_mapping=True,
    grid_size=(5, 7),
    use_interpolation=True,
)
epochs = processor.process()  # list of (T, 5, 7) arrays

# Batch — grid-mapped
dataset = FNIRSDataset(
    root_dir='./data/raw',
    output_dir='./data/processed',
    subject_dict={'healthy': ['AH001', 'AH002'], 'anxiety': ['AA001']},
    task_type='GNG',
    data_type='hbt',
    apply_zscore=True,
    save_preprocessed=True,
    use_grid_mapping=True,
    grid_size=(5, 7),
)
dataset.process()
```
