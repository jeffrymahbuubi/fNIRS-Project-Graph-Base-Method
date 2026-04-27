# Additional Pre-Data Catalog
**Generated:** 2026-04-27  
**NAS Reference:** `references/synology_0331/`  
**Metadata source:** `fNIRS_data20260323.xlsx` (most recent, 34 subjects: AA065–AA099)

---

## Part 1 — Existing `data/raw` Pipeline Source

The processed subjects currently in `data/raw/` originated from:

### Source Location in NAS
```
references/synology_0331/clinical data/rawdata/data_NIRx/rawdata_folders/
```

### Folder Structure (rawdata_folders)
| Task Folder | Sub-category | Recordings | Date Range |
|---|---|---|---|
| `1backWM/` | `anxiety_pre/` | 28 | 2023-06-19 → 2024-05-08 |
| `1backWM/` | `anxiety_post/` | 9 | 2023-08-28 → 2024-02-05 |
| `1backWM/` | `health/` | 36 | 2023-09-04 → 2024-01-04 |
| `SS/` | `anxiety_pre/` | 20 | 2023-10-24 → 2024-05-08 |
| `SS/` | `anxiety_post/` | 4 | 2023-12-27 → 2024-02-05 |
| `SS/` | `health/` | 34 | 2023-12-05 → 2024-01-04 |
| `VF/` | `anxiety_pre/` | 27 | 2023-06-19 → 2024-05-08 |
| `VF/` | `anxiety_post/` | 9 | 2023-08-28 → 2024-02-05 |
| `VF/` | `Health/` | 36 | 2023-09-04 → 2024-01-04 |
| `RP/GNG/` | `anxiety_pre/` | 20 | 2023-10-24 → 2024-05-08 |
| `AP/` | `anxiety_pre/` | 8 | 2023-06-19 → 2023-08-11 |
| `RP/` | `anxiety_pre/` | 8 | 2023-06-19 → 2023-08-11 |

> **Note:** No 2025 date folders exist anywhere in `rawdata_folders`. Latest date is 2024-05-08. All new subjects (AA065+) are outside this folder.

### Metadata Ground Truth
- **File:** `references/synology_0331/clinical data/data_pre.xlsx`
- **Sheet:** `fNIRS` — maps subject IDs → task → date-based file ID
- **Coverage:** 64 subjects total (28 anxiety + 36 healthy, all in `data/raw`)

### Pipeline
```
rawdata_folders/[task]/anxiety_pre/[date_folder]/   →  data/raw/anxiety/[SubjectID]/[task]/
rawdata_folders/[task]/health/[date_folder]/         →  data/raw/healthy/[SubjectID]/[task]/
```
Processed via MATLAB Toolchain at:  
`references/synology_0331/clinical data/rawdata/data_NIRx/Toolchain/`  
(scripts: `nirs2csv.m`, `HomerOfflineConverter_fp.m`, `nirx2homer_batch.m`)

### Existing Subjects in `data/raw`

**Anxiety subjects (28):** AA001–AA008, AA011, AA013, AA041, AA056, AA064, EA012, EA016, EA055, EA060, EA061, EA062, LA042, LA051, LA052, LA053, LA054, LA057, LA058, LA059, LA063

**Healthy subjects (36):** AH009, AH010, AH014–AH050

**Task coverage:**
- AA001–AA008, AH009–AH010: `VF`, `1backWM`, `AP`, `RP` (early protocol)
- All others: `VF`, `1backWM`, `SS`, `GNG`

---

## Part 2 — New Additional "Pre" Data (AA089–AA099)
**Dataset type: No brain stimulation — functional tasks only**

### Source Location in NAS
```
references/synology_0331/clinical data/rawdata/data_NIRx/fNIRS/
```

These are **unprocessed** raw NIRx recordings (`.nirs`, `.snirf`, `.wl1`, `.wl2`).  
No HbO/HbR/HbT CSVs have been generated yet — MATLAB Toolchain processing required before integration.

### Subject Catalog

| Subject ID | Clinical Ref | Tasks | File ID(s) | Synology Raw Path | Notes |
|---|---|---|---|---|---|
| **AA089** | A004 | SS & GNG | `2025-07-02_001` | `fNIRS/2025-07-02/2025-07-02_001/` | |
| **AA090** | — | GNG & SS | `2025-08-10_002` | `fNIRS/2025-08-10/2025-08-10_002/` | |
| **LA091** | — | GNG & SS | `2025-08-28_001` | `fNIRS/2025-08-28/2025-08-28_001/` | |
| **AA092** | A005 | SS & GNG | `2025-09-07_001` | `fNIRS/2025-09-07/2025-09-07_001/` | |
| **AA093** | — | GNG & SS | `2025-09-12_001` | `fNIRS/2025-09-12/2025-09-12_001/` | |
| **AA094** | — | GNG & SS | `2025-09-19_001` | `fNIRS/2025-09-19/2025-09-19_001/` | |
| **LA095** | — | GNG & SS | `2025-09-24_001` | `fNIRS/2025-09-24/2025-09-24_001/` | |
| **LA096** | — | SS & GNG | `2025-12-05_002` | `fNIRS/2025-12-05/2025-12-05_002/` | |
| **AA097** | — | SS & GNG | `2025-12-23_001` + `2025-12-23_002` | `fNIRS/2025-12-23/2025-12-23_001/` + `fNIRS/2025-12-23/2025-12-23_002/` | ⚠️ Split recording — two files for one session, treat as a pair |
| **AA098** | A007 | GNG & SS | `2026-02-26_001` | `fNIRS/2026-02-26/2026-02-26_001/` | |
| **AA099** | A008 | SS & GNG | `2026-03-02_001` | `fNIRS/2026-03-02/2026-03-02_001/` | |

**Total: 11 new subjects** — all absent from `data/raw` as of 2026-04-27.

### Task Coverage — Important Difference
All 11 new subjects have **only 2 tasks** (SS & GNG).  
The original cohort had 4 tasks (VF, 1backWM, SS, GNG).  
**VF and 1backWM recordings do not exist for these subjects.**  
Integration with existing dataset is limited to SS and GNG analyses.

### Raw File Format
Each session folder contains:
```
[date_id]_calibration.json
[date_id]_config.hdr
[date_id]_config.json
[date_id]_description.json
[date_id]_lsl.tri        ← trigger file (confirms valid task recording)
[date_id].nirs
[date_id]_probeInfo.mat
[date_id].snirf
[date_id].wl1
[date_id].wl2
[date_id].zip
digpts.txt
```

---

## Part 3 — Brain Stimulation Cohort (AA065–AH088)
**Dataset type: fNIRS recorded during/alongside brain stimulation (TBS/HD-tES)**  
**For future reference only — separate from non-stimulation dataset**

> These subjects received transcranial brain stimulation (TBS or HD-tES). Their fNIRS data is **not directly comparable** to the original cohort (`data/raw`) or Part 2 subjects, and must be analyzed separately.

### Source Location in NAS
```
references/synology_0331/clinical data/rawdata/data_NIRx/stimulation/
```

### Important: Data Availability in This NAS Backup

The `stimulation/` folder contains the **brain stimulation session recordings** (fNIRS recorded during TBS/HD-tES).  
The **functional task recordings** (GNG, SS, VF, 1backWM) referenced in the xlsx are **separate files** and most are **not present in this NAS backup**.

| Sessions available in `stimulation/` folder | Mapped to subject | Trigger | File size |
|---|---|---|---|
| `2025-04-09_001` | AA065 | ✓ | ~9.3 MB |
| `2025-04-15_001` | ⚠️ Unmatched (LA066 expects `_002`) | ✓ | ~8.0 MB |
| `2025-04-22_001` | ⚠️ Unmatched — no subject in xlsx | ✓ | ~8.1 MB |
| `2025-04-24_001` | LA067 | ✓ | ~9.0 MB |
| `2025-04-29_002` | ⚠️ Unmatched (AH068 expects `_005–_009`) | ✓ | ~8.3 MB |
| `2025-05-01_001` | AH069 (GNG) | ✓ | ~8.1 MB |
| `2025-05-01_002` | AH069 (SS) — workspace copy only | ✗ | ~3.9 MB |
| `2025-05-08_001` | ⚠️ Unmatched — no subject in xlsx | ✓ | ~8.0 MB |
| `2025-05-13_001` | AH072 (GNG) — **0 KB .nirs, likely corrupt** | ✓ | 0 KB |
| `2025-07-02_001` | AA089 (also in Part 2 `fNIRS/` folder) | ✓ | ~8.5 MB |
| `2025-09-07_001` | AA092 (also in Part 2 `fNIRS/` folder) | ✓ | ~8.2 MB |
| `2025-09-14_001` | ⚠️ Unmatched — no subject in xlsx | ✓ | ~8.1 MB |
| `2025-09-19_001` | AA094 (also in Part 2 `fNIRS/` folder) | ✓ | ~8.7 MB |
| `2025-09-24_001` | LA095 (also in Part 2 `fNIRS/` folder) | ✓ | ~8.5 MB |

### Subject Catalog — Stimulation Cohort

| Subject ID | Clinical Ref | Tasks (from xlsx) | Task File IDs | File in `stimulation/` | Note |
|---|---|---|---|---|---|
| **AA065** | A001 | GNG & SS | `2025-04-09_001` | ✓ Present | File doubles as stimulation session |
| **LA066** | — | GNG & SS | `2025-04-15_002` | ✗ Missing | Stim folder has `_001` (different session) |
| **LA067** | L002 | SS & GNG | `2025-04-24_001` | ✓ Present | File doubles as stimulation session |
| **AH068** | — | SS, 1backWM, VF, GNG | `2025-04-29_005–009` | ✗ All missing | Stim folder has `_002` (stimulation only) |
| **AH069** | — | GNG, SS | `2025-05-01_001`, `_002` | `_001` ✓ / `_002` workspace only | `_002` incomplete copy |
| **AH070** | — | 1backWM, GNG, VF, SS | `2025-05-02_001–004` | ✗ All missing | |
| **AH071** | — | VF, SS, GNG | `2025-05-07_003–005` | ✗ All missing | |
| **AH072** | — | GNG, 1backWM, SS, VF | `2025-05-13_001–004` | `_001` present but **0 KB .nirs** | Likely corrupt; `_002–004` missing |
| **AH073** | — | 1backWM, VF, SS, GNG | `2025-05-14_003–006` | ✗ All missing | |
| **AH074** | — | SS, VF, 1backWM, GNG | `2025-05-15_001–004` | ✗ All missing | |
| **AH075** | — | 1backWM, SS, VF, GNG | `2025-05-16_001–004` | ✗ All missing | |
| **AH076** | — | 1backWM, SS, GNG, VF | `2025-05-19_001,003–005` | ✗ All missing | |
| **AH078** | — | GNG, SS, 1backWM, VF | `2025-05-21_001–004` | ✗ All missing | |
| **AH079** | — | SS, GNG, 1backWM, VF | `2025-05-22_001–004` | ✗ All missing | |
| **AH080** | — | VF, 1backWM, GNG, SS | `2025-05-22_005–008` | ✗ All missing | |
| **AH081** | — | 1backWM, GNG, SS, VF | `2025-05-22_010–013` | ✗ All missing | |
| **AH082** | — | SS, VF, GNG, 1backWM | `2025-05-23_001–004` | ✗ All missing | |
| **AH083** | — | SS, GNG, VF, 1backWM | `2025-05-23_005–008` | ✗ All missing | |
| **AH084** | — | GNG, 1backWM, VF, SS | `2025-05-23_009–012` | ✗ All missing | |
| **AH085** | — | VF, 1backWM, SS, GNG | `2025-05-26_002–005` | ✗ All missing | |
| **AH086** | — | VF, GNG, SS, 1backWM | `2025-05-28_001–004` | ✗ All missing | |
| **AH087** | — | GNG, VF, SS, 1backWM | `2025-05-28_005–008` | ✗ All missing | |
| **AH088** | — | VF, SS, 1backWM, GNG | `2025-06-02_001–004` | ✗ All missing | |

**Total stimulation cohort: 23 subjects** (AA065–AH088)

### Data Completeness Summary
- Functional task recordings present in NAS: **AA065, LA067** (2/23 complete)
- Partial / uncertain: **AH069** (1 of 2 tasks), **AH072** (corrupt file)
- Functional task recordings missing from NAS: **19 subjects** (AH068, AH070–AH088 and LA066)
- Stimulation session recordings with no subject match: 5 sessions (`2025-04-15_001`, `2025-04-22_001`, `2025-04-29_002`, `2025-05-08_001`, `2025-09-14_001`)

> **Note on dual-folder subjects (AA089, AA092, AA094, LA095):** These subjects appear in BOTH `stimulation/` and `fNIRS/` folders. Their functional task data is cataloged in Part 2. Their stimulation session data is here in `stimulation/` as a separate recording.

---

## Part 4 — Excluded Recordings

### Unaccounted Recording — `fNIRS/2025-12-05_001`
Exists at `fNIRS/2025-12-05/2025-12-05_001/` but NOT mapped to any subject in `fNIRS_data20260323.xlsx`.  
File size: ~596 KB (`.nirs` = 58 KB, no `_lsl.tri` trigger file).  
Compare: LA096's `2025-12-05_002` is 62 MB with trigger file.  
**Conclusion:** Failed/aborted calibration attempt before LA096's session. **Do not process.**

---

## Part 5 — Integration Checklist (Future)

### For Part 2 subjects (non-stimulation, AA089–AA099)
- [ ] Run MATLAB Toolchain on each `fNIRS/[date]/[file_id]/` folder → HbO/HbR/HbT CSVs
- [ ] Organize output into `data/raw/anxiety/[SubjectID]/[task]/` (SS and GNG only)
- [ ] Handle AA097 split recording (`_001` + `_002`) — concatenate before processing
- [ ] Update `data/subjects.json` with new subject entries
- [ ] Verify `_lsl.tri` trigger files are present for all 11 recordings before processing

### For Part 3 subjects (stimulation cohort, AA065–AH088)
- [ ] Retrieve missing functional task recordings from the actual NAS device (not in this backup)
- [ ] Keep stimulation session recordings (`stimulation/`) in a separate analysis pipeline
- [ ] Do NOT merge stimulation-session fNIRS data with the non-stimulation dataset
- [ ] AH072's `2025-05-13_001` is likely corrupt (0 KB .nirs) — verify on source NAS
