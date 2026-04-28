import mne
from mne import create_info
import pandas as pd
import numpy as np
import copy
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import argparse
import json
import warnings
from scipy.interpolate import Rbf

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure module-level logger
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FNIRSDataProcessor:
    """
    Process fNIRS data for specific subjects, groups, and tasks.

    Parameters:
    - root_dir: Path to directory containing 'healthy' and 'anxiety' folders
    - subject: Subject identifier (string)
    - group: 'healthy' or 'anxiety'
    - task_type: 'GNG', '1backWM', 'VF', or 'SS'
    - data_type: 'hbo', 'hbr', or 'hbt'
    - apply_baseline: bool, whether to apply baseline correction
    - apply_zscore: bool, whether to apply z-score normalization
    - save_preprocessed: bool, whether to save each epoch
    - save_format: 'npy' or 'txt', format for saving preprocessed data
    - montage_file: path to custom montage file (optional)
    - ppf: partial pathlength factor for Beer-Lambert law
    """
    def __init__(
        self,
        root_dir,
        subject,
        group,
        task_type='GNG',
        data_type='hbo',
        apply_baseline=False,
        apply_zscore=False,
        save_preprocessed=False,
        save_format='npy',
        montage_file=None,
        ppf=6.0,
        use_grid_mapping=False,
        grid_size=(5, 7),
        use_interpolation=True,
        save_plots=False,
        plot_dir=None,
    ):
        self.root_dir = Path(root_dir)
        self.subject = subject
        self.group = group
        self.task_type = task_type
        self.data_type = data_type.lower()
        self.apply_baseline = apply_baseline
        self.apply_zscore = apply_zscore
        self.save_preprocessed = save_preprocessed
        self.save_format = save_format.lower()
        self.montage_file = montage_file
        self.ppf = ppf
        self.use_grid_mapping = use_grid_mapping
        self.grid_size = grid_size
        self.use_interpolation = use_interpolation
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir) if plot_dir else None

        # Define subject directory: root_dir / group / subject / task
        self.subject_dir = self.root_dir / self.group / self.subject / self.task_type

        # Placeholders
        self.raw_haemo = None
        self.preprocessed = None
        self.epoch_data = None

    def load_data(self):
        """
        Load raw NIRx data, apply OD and Beer-Lambert, and combine HbO/HbR CSVs.
        """
        # Read raw data
        raw = mne.io.read_raw_nirx(self.subject_dir, preload=True, verbose=False)
        # Set custom montage if provided
        if self.montage_file:
            montage = mne.channels.read_custom_montage(self.montage_file, head_size=0.0825)
            raw.set_montage(montage)

        # Convert to optical density and hemoglobin concentration
        raw_od = mne.preprocessing.nirs.optical_density(raw, verbose=False)
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=self.ppf)

        # Load preprocessed CSV files
        hbo_file = next(self.subject_dir.glob('*HbO.csv'), None)
        hbr_file = next(self.subject_dir.glob('*HbR.csv'), None)
        hbo_raw = pd.read_csv(hbo_file, header=None)
        hbr_raw = pd.read_csv(hbr_file, header=None)

        # Row 0 is the time vector; its first value is the CSV window start in seconds.
        # Task-split CSVs (from split_combined_sessions.m) keep original timestamps, so
        # t_offset > 0 for whichever task occupies the second half of a combined session.
        t_offset = float(hbo_raw.iloc[0, 0])

        hbo_df = hbo_raw.drop([0, 1]).reset_index(drop=True).astype(float)
        hbr_df = hbr_raw.drop([0, 1]).reset_index(drop=True).astype(float)
        assert hbo_df.shape[0] == hbr_df.shape[0], "HbO and HbR rows must match"

        # Interleave rows: HbO, HbR
        combined = pd.DataFrame(np.empty((hbo_df.shape[0] * 2, hbo_df.shape[1])))
        combined.iloc[0::2, :] = hbo_df.values
        combined.iloc[1::2, :] = hbr_df.values

        # Create RawArray and align annotations to the CSV time window.
        # Shifting by t_offset maps original-recording timestamps onto the 0-indexed
        # RawArray time axis.  Events outside the window (other task's triggers) become
        # negative or exceed signal_duration and are filtered out.
        preprocessed = mne.io.RawArray(combined.values, raw_haemo.info)
        signal_duration = combined.shape[1] / raw_haemo.info['sfreq']
        ann = raw_haemo.annotations
        shifted_onsets = ann.onset - t_offset
        valid = (shifted_onsets >= 0) & (shifted_onsets < signal_duration)
        adjusted_ann = mne.Annotations(
            onset=shifted_onsets[valid],
            duration=ann.duration[valid],
            description=np.array(ann.description)[valid],
            orig_time=ann.orig_time,
        )
        preprocessed.set_annotations(adjusted_ann)

        self.raw_haemo = raw_haemo
        self.preprocessed = preprocessed
        return raw_haemo, preprocessed

    def generate_hbt(self):
        """
        Compute HbT channel as HbO + HbR and add to preprocessed data.
        """
        hbo = self.preprocessed.get_data(picks='hbo')
        hbr = self.preprocessed.get_data(picks='hbr')
        hbt = hbo + hbr

        names = [ch.replace('hbo', 'hbt') for ch in self.raw_haemo.ch_names if 'hbo' in ch]
        info = create_info(names, self.preprocessed.info['sfreq'], ch_types='fnirs_cw_amplitude')
        raw_hbt = mne.io.RawArray(hbt, info)
        self.preprocessed.add_channels([raw_hbt])

    def modify_annotations(self):
        """
        Update annotation durations and insert 'Rest' after first baseline.
        """
        ann = copy.deepcopy(self.preprocessed.annotations)
        correct = {'1.0': 120}
        new_on, new_dur, new_desc = [], [], []
        rest_added = False

        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
            new_dur.append(correct.get(desc, duration))
            new_on.append(onset)
            new_desc.append(desc)

            if desc == '1.0' and not rest_added:
                new_on.append(onset + correct['1.0'])
                new_dur.append(30)
                new_desc.append('99.0')
                rest_added = True

        updated = mne.Annotations(onset=new_on, duration=new_dur, description=new_desc, orig_time=ann.orig_time)
        self.preprocessed.set_annotations(updated)

    def save_metadata(self, save_dir):
        """
        Save the subject metadata to a file in the following format:

        [name]
        subject_id

        [metadata]
        sfreq

        The file is saved in the same folder as the time marker plot
        (i.e. self.output_dir / group / subject) unless an alternative output_dir is specified.

        Parameters:
            save_dir (str): The directory where the metadata file will be saved.
        """
        if save_dir is None:
            raise ValueError("save_dir must be specified to save metadata.")

        raw = mne.io.read_raw_nirx(self.subject_dir, preload=True, verbose=False)

        save_dir = Path(save_dir)

        sfreq = raw.info['sfreq']

        file_path = save_dir / f"{self.subject}.data"
        file_content = f"[name]\nsubject_id={self.subject}\n\n[metadata]\nsfreq={sfreq}\n"

        with open(file_path, 'w') as f:
            f.write(file_content)

    def _baseline_correction(self, epochs):
        """Baseline correct using first Rest epoch (99.0)."""
        rest = epochs['99.0'].get_data()[0]
        idx = np.where((epochs.times >= 0) & (epochs.times <= 5))[0]
        mean = rest[:, idx].mean(axis=1)
        data = epochs.get_data()
        data -= mean[:, np.newaxis]
        epochs._data = data
        return epochs

    def plot_time_marker(self, save_dir):
        """Save event time marker plot for the subject."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        events, event_id = mne.events_from_annotations(self.preprocessed, verbose=False)
        if len(events) == 0:
            logger.warning(f"No events found for {self.subject}, skipping time marker plot.")
            return
        fig = mne.viz.plot_events(
            events, event_id=event_id,
            sfreq=self.preprocessed.info['sfreq'], show=False
        )
        fig.suptitle(f"{self.task_type} Task Time Marker for {self.subject}", fontsize=16)
        fig.savefig(save_dir / f"{self.task_type}_time_marker.png")
        plt.close(fig)

    def plot_evoked(self, save_dir):
        """Save evoked response plot (HbO/HbR/HbT, Baseline vs Task) for the subject."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        data = self.preprocessed.copy()
        present = set(data.annotations.description)
        if self.task_type in ('SS', 'VF'):
            _rename = {"1.0": "Baseline", "3.0": "Task", "4.0": "Task", "5.0": "Task", "6.0": "Task"}
        else:
            _rename = {"1.0": "Baseline", "3.0": "Task", "4.0": "Task"}
        filtered_rename = {k: v for k, v in _rename.items() if k in present}
        if filtered_rename:
            data.annotations.rename(filtered_rename)

        task_durations = {'GNG': 35, 'VF': 60, 'SS': 60, '1backWM': 90}
        tmax = task_durations.get(self.task_type, 35)

        events, event_id = mne.events_from_annotations(data, verbose=False)
        if 'Baseline' not in event_id or 'Task' not in event_id:
            logger.warning(f"Missing Baseline or Task events for {self.subject}, skipping evoked plot.")
            return

        # Temporarily add HbT for plotting if not present
        if not any('hbt' in ch for ch in data.ch_names):
            hbt = data.get_data(picks='hbo') + data.get_data(picks='hbr')
            hbt_names = [ch.replace('hbo', 'hbt') for ch in self.raw_haemo.ch_names if 'hbo' in ch]
            hbt_info = create_info(hbt_names, data.info['sfreq'], ch_types='fnirs_cw_amplitude')
            data.add_channels([mne.io.RawArray(hbt, hbt_info)])

        epochs = mne.Epochs(
            data, events, event_id, tmin=-5, tmax=tmax,
            reject_by_annotation=True, proj=True, baseline=None,
            preload=True, detrend=None, event_repeated='drop', verbose=False
        )
        epochs = copy.deepcopy(epochs).apply_baseline(baseline=(-5, 0), verbose=False)
        epochs.crop(tmin=0, tmax=epochs.tmax)

        picks_hbt = [ch for ch in epochs.ch_names if 'hbt' in ch]
        evoked_dict = {
            "Baseline/HbO": epochs["Baseline"].average(picks="hbo"),
            "Baseline/HbR": epochs["Baseline"].average(picks="hbr"),
            "Baseline/HbT": epochs["Baseline"].average(picks=picks_hbt),
            "Task/HbO": epochs["Task"].average(picks="hbo"),
            "Task/HbR": epochs["Task"].average(picks="hbr"),
            "Task/HbT": epochs["Task"].average(picks=picks_hbt),
        }
        for cond in evoked_dict:
            evoked_dict[cond].rename_channels(lambda x: x[:-4] if " " in x else x)

        color_dict = dict(HbO="r", HbR="b", HbT="g")
        styles_dict = {
            "Baseline/HbO": dict(linestyle="dashed"),
            "Baseline/HbR": dict(linestyle="dashed"),
            "Baseline/HbT": dict(linestyle="dashed"),
        }
        fig = mne.viz.plot_compare_evokeds(
            evoked_dict, combine="mean", ci=0.95,
            colors=color_dict, styles=styles_dict, show=False
        )[0]
        fig.suptitle(f"{self.task_type} Task Evoked for {self.subject}", fontsize=16)
        fig.savefig(save_dir / f"{self.task_type}_evoked.png")
        plt.close(fig)

    def get_channel_positions(self, signal_type):
        """Return the 23-channel → 5×7 grid position mapping."""
        return {
            f'S1_D1 {signal_type}': (0, 2), f'S1_D3 {signal_type}': (1, 1), f'S2_D2 {signal_type}': (0, 4),
            f'S2_D1 {signal_type}': (0, 3), f'S2_D5 {signal_type}': (1, 4), f'S3_D1 {signal_type}': (1, 2),
            f'S3_D3 {signal_type}': (2, 1), f'S3_D4 {signal_type}': (2, 2), f'S3_D6 {signal_type}': (3, 1),
            f'S4_D4 {signal_type}': (2, 3), f'S4_D5 {signal_type}': (2, 4), f'S4_D7 {signal_type}': (3, 4),
            f'S5_D2 {signal_type}': (1, 5), f'S5_D5 {signal_type}': (2, 5), f'S5_D8 {signal_type}': (3, 6),
            f'S6_D3 {signal_type}': (3, 0), f'S6_D6 {signal_type}': (4, 1), f'S7_D4 {signal_type}': (3, 2),
            f'S7_D6 {signal_type}': (4, 2), f'S7_D7 {signal_type}': (4, 3), f'S8_D5 {signal_type}': (3, 5),
            f'S8_D7 {signal_type}': (4, 4), f'S8_D8 {signal_type}': (4, 5),
        }

    def interpolate_missing(self, matrix):
        """Fill zero-valued grid cells via Gaussian RBF interpolation."""
        x_known, y_known = np.where(matrix != 0)
        values_known = matrix[matrix != 0]
        x_missing, y_missing = np.where(matrix == 0)
        if len(values_known) > 0 and len(x_missing) > 0:
            rbf = Rbf(x_known, y_known, values_known, function='gaussian')
            matrix[x_missing, y_missing] = rbf(x_missing, y_missing)
        return matrix

    def generate_matrix(self, data, channel_positions):
        """Convert epoch (23, T) to grid sequence (T, H, W)."""
        n_timepoints = data.shape[1]
        grids = []
        for t in range(n_timepoints):
            grid = np.full(self.grid_size, 0.0)
            for ch_idx, (row, col) in enumerate(channel_positions.values()):
                if ch_idx < data.shape[0]:
                    grid[row, col] = data[ch_idx, t]
            if self.use_interpolation:
                grid = self.interpolate_missing(grid)
            grids.append(np.nan_to_num(grid))
        return np.stack(grids, axis=0)  # (T, H, W)

    def epoch(self):
        """
        Epoch data around events, apply optional baseline and z-score, and save.
        Returns a list of numpy arrays for each epoch.
        """
        durations = {
            'GNG': (0, 35),
            'VF': (0, 60),
            'SS': (0, 60),
            '1backWM': (0, 90)
        }
        tmin, tmax = durations.get(self.task_type, (0, 35))

        # Select picks
        if self.data_type in ['hbo', 'hbr']:
            picks = self.data_type
        else:
            picks = [ch for ch in self.preprocessed.info['ch_names'] if 'hbt' in ch]

        events, event_id = mne.events_from_annotations(self.preprocessed, verbose=False)
        epochs = mne.Epochs(
            self.preprocessed.copy().pick(picks),
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            reject_by_annotation=True,
            proj=True,
            baseline=None,
            preload=True,
            detrend=None,
            event_repeated='drop',
            verbose=False
        )

        if self.apply_baseline:
            epochs = self._baseline_correction(epochs)

        # Select only task events
        _MULTI_EVENTS = {'3.0', '4.0', '5.0', '6.0'}
        if self.task_type in ('SS', 'VF'):
            task_evs = [k for k in epochs.event_id if k in _MULTI_EVENTS]
            if task_evs:
                # Old SS/VF protocol: codes 3–6 are four distinct conditions (code 6 always present).
                # New protocol: code 3 or 4 = SS trials (×4), code 5 = session/section marker (×1).
                # Exclude code 5 when code 6 is absent — it is a recording marker, not a condition.
                if '6.0' not in epochs.event_id:
                    task_evs = [k for k in task_evs if k != '5.0']
                if task_evs:
                    epochs = epochs[task_evs]
        elif '3.0' in epochs.event_id:
            epochs = epochs['3.0']
        elif '4.0' in epochs.event_id:
            # Combined-session GNG: task events have code 4; code 3 (SS) was
            # filtered out by load_data because it falls outside this CSV window.
            epochs = epochs['4.0']

        # Crop per-task preparation window before z-score so stats reflect only task data
        _PREP_CROP = {'GNG': 3, 'SS': 7, '1backWM': 5, 'VF': 7}
        tmin_crop = _PREP_CROP.get(self.task_type, 3)
        epochs = epochs.crop(tmin=tmin_crop, tmax=epochs.tmax)

        if self.apply_zscore:
            def z(ts): return (ts - ts.mean()) / (ts.std() + 1e-8)
            epochs.apply_function(z, picks=None, channel_wise=True, verbose=False)

        channel_positions = self.get_channel_positions(self.data_type) if self.use_grid_mapping else None

        self.epoch_data = []
        for idx in range(len(epochs)):
            arr = epochs[idx].get_data().squeeze()  # (23, T)
            if self.use_grid_mapping:
                arr = self.generate_matrix(arr, channel_positions)  # (T, H, W)
            self.epoch_data.append(arr)

            if self.save_preprocessed:
                out_dir = self.subject_dir / self.data_type.upper()
                out_dir.mkdir(parents=True, exist_ok=True)

                if self.save_format == 'txt':
                    np.savetxt(out_dir / f"{idx}.txt", arr)
                else:  # default to npy
                    np.save(out_dir / f"{idx}.npy", arr)

        return self.epoch_data

    def process(self):
        """
        Execute full pipeline: load, optional HbT, annotate, epoch.
        Returns list of epoch arrays.
        """
        self.load_data()
        if self.data_type == 'hbt':
            self.generate_hbt()
        self.modify_annotations()
        if self.save_plots:
            plot_save_dir = self.plot_dir if self.plot_dir else self.subject_dir.parent
            self.plot_time_marker(plot_save_dir)
            self.plot_evoked(plot_save_dir)
        return self.epoch()

class FNIRSDataset:
    """
    Process multiple subjects collectively and optionally save preprocessed epochs with logging.
    """
    def __init__(
        self,
        root_dir,
        output_dir,
        subject_dict,
        task_type='GNG',
        data_type='hbo',
        apply_baseline=False,
        apply_zscore=False,
        save_preprocessed=False,
        save_format='npy',
        montage_file=None,
        ppf=6.0,
        use_grid_mapping=False,
        grid_size=(5, 7),
        use_interpolation=True,
        save_plots=False,
    ):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.subject_dict = subject_dict
        self.task_type = task_type
        self.data_type = data_type.lower()
        self.apply_baseline = apply_baseline
        self.apply_zscore = apply_zscore
        self.save_preprocessed = save_preprocessed
        self.save_format = save_format.lower()
        self.montage_file = montage_file
        self.ppf = ppf
        self.use_grid_mapping = use_grid_mapping
        self.grid_size = grid_size
        self.use_interpolation = use_interpolation
        self.save_plots = save_plots
        self.logger = logging.getLogger(f"{__name__}.FNIRSDataset")
        self.logger.info(f"FNIRSDataset initialized: task_type={self.task_type}, data_type={self.data_type}, "
                         f"apply_baseline={self.apply_baseline}, apply_zscore={self.apply_zscore}, "
                         f"save_preprocessed={self.save_preprocessed}, "
                         f"use_grid_mapping={self.use_grid_mapping}, grid_size={self.grid_size}")

    def process(self):
        self.logger.info(f"Starting dataset processing for task '{self.task_type}' and data type '{self.data_type}'")
        for group, subjects in self.subject_dict.items():
            self.logger.info(f"Processing group '{group}' with {len(subjects)} subjects: {subjects}")
            for subj in subjects:
                self.logger.info(f"Processing subject '{subj}' in group '{group}'")
                try:
                    plot_dir = self.output_dir / self.task_type / group / subj
                    proc = FNIRSDataProcessor(
                        self.root_dir,
                        subj,
                        group,
                        task_type=self.task_type,
                        data_type=self.data_type,
                        apply_baseline=self.apply_baseline,
                        apply_zscore=self.apply_zscore,
                        save_preprocessed=False,
                        montage_file=self.montage_file,
                        ppf=self.ppf,
                        use_grid_mapping=self.use_grid_mapping,
                        grid_size=self.grid_size,
                        use_interpolation=self.use_interpolation,
                        save_plots=self.save_plots,
                        plot_dir=plot_dir,
                    )

                    base = self.output_dir / self.task_type / group / subj / self.data_type
                    base.mkdir(parents=True, exist_ok=True)

                    epoch_list = proc.process()
                    self.logger.info(f"Subject '{subj}' processed: {len(epoch_list)} epochs generated.")

                    # 2) save the metadata alongside each subject
                    subj_dir = self.output_dir / self.task_type / group / subj
                    proc.save_metadata(subj_dir)
                    self.logger.info(f"Metadata saved for subject '{subj}'.")

                    if self.save_preprocessed:
                        self.logger.info(f"Saving epochs for subject '{subj}' to {base} in {self.save_format} format")
                        for idx, arr in enumerate(epoch_list):
                            if self.save_format == 'txt':
                                file_path = base / f"{idx}.txt"
                                np.savetxt(file_path, arr)
                                self.logger.debug(f"Saved epoch {idx}.txt to {file_path}")
                            else:  # default to npy
                                file_path = base / f"{idx}.npy"
                                np.save(file_path, arr)
                                self.logger.debug(f"Saved epoch {idx}.npy to {file_path}")

                except Exception as e:
                    self.logger.error(f"Failed processing subject '{subj}' in group '{group}': {e}", exc_info=True)

        self.logger.info("Processing complete for all subjects.")

def validate_data_type(output_dir, subject):
    """
    Load all epochs for a subject across HbO, HbR, and HbT,
    compute the mean signal across channels and epochs at each timepoint,
    and plot them in one figure (HBO-red, HBR-blue, HBT-green).

    Supports both .npy and .txt file formats.

    Parameters:
    - output_dir: root directory where processed data is saved (including task_type/group/...)
    - subject: subject ID (str)
    """
    output_dir = Path(output_dir)
    data_types = ['hbo', 'hbr', 'hbt']
    mean_signals = {}

    for dtype in data_types:
        # Find all .npy or .txt files for this subject and dtype
        npy_files = list(output_dir.rglob(f"{subject}/{dtype}/*.npy"))
        txt_files = list(output_dir.rglob(f"{subject}/{dtype}/*.txt"))

        if npy_files:
            files = npy_files
            load_func = np.load
        elif txt_files:
            files = txt_files
            load_func = np.loadtxt
        else:
            logger.warning(f"No files found for subject={subject}, dtype={dtype}")
            continue

        # Load arrays: shape per file (channels, timepoints)
        arrays = [load_func(f) for f in sorted(files)]
        data = np.stack(arrays)  # (n_epochs, channels, timepoints)
        # Compute mean over epochs and channels
        mean_signal = data.mean(axis=(0, 1))  # shape (timepoints,)
        mean_signals[dtype] = mean_signal

    # Plot
    plt.figure(figsize=(8, 5))
    for dtype, signal in mean_signals.items():
        color = {'hbo': 'r', 'hbr': 'b', 'hbt': 'g'}.get(dtype, 'k')
        plt.plot(signal, color=color, label=dtype.upper())
    plt.xlabel('Time (samples)', fontname="Times New Roman", fontsize=14, labelpad=10)
    plt.ylabel('Mean signal', fontname="Times New Roman", fontsize=14, labelpad=10)
    plt.title(f'Average fNIRS signals for subject {subject}',
              fontname="Times New Roman", fontweight="bold", fontsize=16, pad=20)
    plt.legend(prop={"family": "Times New Roman", "size": 12})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def parse_arguments():
    """
    Parse command-line arguments for fNIRS data processing.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Process fNIRS RAW data files with flexible configuration options.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject
  python processor.py --mode single --root-dir ./data/raw_data --subject AH001 --group healthy --task GNG --data-type hbo

  # Process multiple subjects with baseline correction and z-score
  python processor.py --mode batch --root-dir ./data/raw_data --output-dir ./data/processed_data \\
    --subjects-json subjects.json --task GNG --data-type hbo --apply-baseline --apply-zscore --save-preprocessed

  # Validate processed data
  python processor.py --mode validate --output-dir ./data/processed_data --subject AH001

Example subjects.json format:
  {
    "healthy": ["AH001", "AH002", "AH003"],
    "anxiety": ["AA001", "AA002"]
  }
        """
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['single', 'batch', 'validate'],
        help='Processing mode: single (one subject), batch (multiple subjects), validate (check processed data)'
    )

    # Common arguments
    parser.add_argument(
        '--root-dir',
        type=str,
        help='Path to root directory containing raw data (required for single and batch modes)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Path to output directory for processed data (required for batch and validate modes)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='GNG',
        choices=['GNG', '1backWM', 'VF', 'SS'],
        help='Task type (default: GNG)'
    )

    parser.add_argument(
        '--data-type',
        type=str,
        default='hbo',
        choices=['hbo', 'hbr', 'hbt'],
        help='Data type to process (default: hbo)'
    )

    # Single mode arguments
    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID (required for single and validate modes)'
    )

    parser.add_argument(
        '--group',
        type=str,
        choices=['healthy', 'anxiety'],
        help='Subject group (required for single mode)'
    )

    # Batch mode arguments
    parser.add_argument(
        '--subjects-json',
        type=str,
        help='Path to JSON file containing subject dictionary (required for batch mode). Format: {"group1": ["subj1", "subj2"], "group2": [...]}'
    )

    # Processing options
    parser.add_argument(
        '--apply-baseline',
        action='store_true',
        help='Apply baseline correction'
    )

    parser.add_argument(
        '--apply-zscore',
        action='store_true',
        help='Apply z-score normalization'
    )

    parser.add_argument(
        '--save-preprocessed',
        action='store_true',
        help='Save preprocessed epoch data'
    )

    parser.add_argument(
        '--save-format',
        type=str,
        default='npy',
        choices=['npy', 'txt'],
        help='Format for saving preprocessed data (default: npy)'
    )

    parser.add_argument(
        '--montage-file',
        type=str,
        default='brainproducts-RNP-BA-128-custom.elc',
        help='Path to custom montage file (optional)'
    )

    parser.add_argument(
        '--ppf',
        type=float,
        default=6.0,
        help='Partial pathlength factor for Beer-Lambert law (default: 6.0)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--use-grid-mapping',
        action='store_true',
        help='Apply 5×7 channel-to-grid mapping with Gaussian RBF interpolation. '
             'Saves epochs as (T, H, W) instead of (23, T).'
    )

    parser.add_argument(
        '--grid-size',
        type=str,
        default='5,7',
        help='Grid dimensions as H,W (default: 5,7). Only used when --use-grid-mapping is set.'
    )

    parser.add_argument(
        '--no-interpolation',
        action='store_true',
        help='Disable Gaussian RBF interpolation for empty grid cells (only with --use-grid-mapping)'
    )

    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save *_time_marker.png and *_evoked.png plots per subject to the output directory'
    )

    return parser.parse_args()


def validate_args(args):
    """
    Validate argument combinations based on mode.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If required arguments are missing for the selected mode
    """
    if args.mode == 'single':
        if not args.root_dir:
            raise ValueError("--root-dir is required for single mode")
        if not args.subject:
            raise ValueError("--subject is required for single mode")
        if not args.group:
            raise ValueError("--group is required for single mode")

    elif args.mode == 'batch':
        if not args.root_dir:
            raise ValueError("--root-dir is required for batch mode")
        if not args.output_dir:
            raise ValueError("--output-dir is required for batch mode")
        if not args.subjects_json:
            raise ValueError("--subjects-json is required for batch mode")
        if not Path(args.subjects_json).exists():
            raise ValueError(f"Subjects JSON file not found: {args.subjects_json}")

    elif args.mode == 'validate':
        if not args.output_dir:
            raise ValueError("--output-dir is required for validate mode")
        if not args.subject:
            raise ValueError("--subject is required for validate mode")


def main():
    """
    Main entry point for command-line processing of fNIRS data.
    """
    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Argument validation failed: {e}")
        return 1

    grid_size = tuple(int(x) for x in args.grid_size.split(','))
    use_interpolation = not args.no_interpolation

    try:
        if args.mode == 'single':
            logger.info(f"Processing single subject: {args.subject} (group: {args.group}, task: {args.task})")

            plot_dir = Path(args.output_dir) / args.task / args.group / args.subject if args.output_dir else None
            processor = FNIRSDataProcessor(
                root_dir=args.root_dir,
                subject=args.subject,
                group=args.group,
                task_type=args.task,
                data_type=args.data_type,
                apply_baseline=args.apply_baseline,
                apply_zscore=args.apply_zscore,
                save_preprocessed=args.save_preprocessed,
                save_format=args.save_format,
                montage_file=args.montage_file,
                ppf=args.ppf,
                use_grid_mapping=args.use_grid_mapping,
                grid_size=grid_size,
                use_interpolation=use_interpolation,
                save_plots=args.save_plots,
                plot_dir=plot_dir,
            )

            epoch_data = processor.process()
            logger.info(f"Successfully processed {len(epoch_data)} epochs for subject {args.subject}")

            # Save metadata if output_dir is provided
            if args.output_dir:
                output_path = Path(args.output_dir) / args.task / args.group / args.subject
                output_path.mkdir(parents=True, exist_ok=True)
                processor.save_metadata(output_path)
                logger.info(f"Metadata saved to {output_path}")

        elif args.mode == 'batch':
            logger.info(f"Processing batch mode with subjects from {args.subjects_json}")

            # Load subjects dictionary
            with open(args.subjects_json, 'r') as f:
                subject_dict = json.load(f)

            logger.info(f"Loaded subjects: {subject_dict}")

            dataset = FNIRSDataset(
                root_dir=args.root_dir,
                output_dir=args.output_dir,
                subject_dict=subject_dict,
                task_type=args.task,
                data_type=args.data_type,
                apply_baseline=args.apply_baseline,
                apply_zscore=args.apply_zscore,
                save_preprocessed=args.save_preprocessed,
                save_format=args.save_format,
                montage_file=args.montage_file,
                ppf=args.ppf,
                use_grid_mapping=args.use_grid_mapping,
                grid_size=grid_size,
                use_interpolation=use_interpolation,
                save_plots=args.save_plots,
            )

            dataset.process()
            logger.info("Batch processing completed successfully")

        elif args.mode == 'validate':
            logger.info(f"Validating processed data for subject {args.subject}")
            validate_data_type(args.output_dir, args.subject)
            logger.info("Validation plot displayed")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
    plt.show()