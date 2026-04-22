# fNIRS Data Processor - Command Line Interface

This document describes how to use the `processor.py` script with command-line arguments for processing fNIRS RAW data files.

## Features

The refactored processor now supports:
- **Single subject processing**: Process one subject at a time
- **Batch processing**: Process multiple subjects from a JSON configuration
- **Data validation**: Visualize processed data for quality control
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
python processor.py \
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
python processor.py \
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
python processor.py \
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

### Logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging verbosity (default: INFO)

## Examples

### Example 1: Quick single subject processing
```bash
python processor.py --mode single --root-dir ./data/raw_data --subject AH001 --group healthy
```

### Example 2: Batch processing with all preprocessing steps
```bash
python processor.py \
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
python processor.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --data-type hbt \
  --apply-baseline
```

### Example 4: Validate processed data
```bash
python processor.py --mode validate --output-dir ./data/processed_data --subject AH001
```

### Example 5: Custom montage and PPF
```bash
python processor.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --montage-file ./misc/brainproducts-RNP-BA-128-custom.elc \
  --ppf 7.5
```

### Example 6: Save preprocessed data as text files
```bash
python processor.py \
  --mode single \
  --root-dir ./data/raw_data \
  --subject AH001 \
  --group healthy \
  --save-preprocessed \
  --save-format txt
```

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

Output structure (batch mode with --save-preprocessed):
```
output_dir/
└── GNG/
    ├── healthy/
    │   └── AH001/
    │       ├── AH001.data (metadata)
    │       └── hbo/
    │           ├── 0.npy (or 0.txt with --save-format txt)
    │           ├── 1.npy (or 1.txt with --save-format txt)
    │           └── ...
    └── anxiety/
        └── AA001/
```

## Tips

1. **Start with single mode** to test parameters on one subject before batch processing
2. **Use --log-level DEBUG** for detailed troubleshooting
3. **Validate your data** after processing to ensure quality
4. **Create a subjects.json** template for reproducible batch processing
5. **Choose save format wisely**: Use `npy` for faster loading in Python/NumPy, or `txt` for human-readable format and compatibility with other tools
6. **Save preprocessed data** only when needed to save disk space

## Programmatic Usage

The classes can still be imported and used in Python scripts:

```python
from processor import FNIRSDataProcessor, FNIRSDataset

# Single subject
processor = FNIRSDataProcessor(
    root_dir='./data/raw_data',
    subject='AH001',
    group='healthy',
    task_type='GNG',
    data_type='hbo',
    apply_baseline=True,
    save_preprocessed=True,
    save_format='txt'  # or 'npy'
)
epochs = processor.process()

# Multiple subjects
dataset = FNIRSDataset(
    root_dir='./data/raw_data',
    output_dir='./data/processed_data',
    subject_dict={'healthy': ['AH001', 'AH002']},
    task_type='GNG',
    data_type='hbo',
    save_preprocessed=True,
    save_format='txt'  # or 'npy'
)
dataset.process()
```
