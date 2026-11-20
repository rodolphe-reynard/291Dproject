from pathlib import Path

import numpy as np

# --- Paths
DATA_DIR = Path("C:/Users/Home/Code/291Dproject/data-z24")  # TODO replace with your path
OUTPUT_DIR = Path("outputs")  # generated files (caches)

# --- Scenarios
SCENARIOS = list(range(1, 18))

SEVERITY = {  # assumed values, not measurements
    1: 0.0, 2: 0.0, 8: 0.0,
    3: 0.15, 4: 0.28, 5: 0.43, 6: 0.60, 7: 0.75,
    9: 0.18, 10: 0.30, 11: 0.42, 12: 0.55, 13: 0.62, 14: 0.70, 15: 0.78, 16: 0.88, 17: 1.0,
}

DATES = {  # PDT test day of each scenario (yyyy/mm/dd)
    1: "1998/08/04", 2: "1998/08/09", 3: "1998/08/11", 4: "1998/08/13", 5: "1998/08/17",
    6: "1998/08/18", 7: "1998/08/19", 8: "1998/08/20", 9: "1998/08/25", 10: "1998/08/26",
    11: "1998/08/27", 12: "1998/08/31", 13: "1998/09/02", 14: "1998/09/03", 15: "1998/09/04",
    16: "1998/09/08", 17: "1998/09/09",
}

# --- EMS (one year of measurements)
EMS_CHANNELS = [3, 5, 6, 7, 10, 12, 14, 16]
DAMAGE_START = "1998/08/04"  # EMS is healthy before this day

# --- Spectra
FS = 100.0  # sampling frequency (Hz), same for PDT and EMS
N_PERSEG = 4096  # Welch segment: 0.024 Hz resolution
F_MIN, F_MAX = (1.0, 25.0)  # frequency range of interest (Hz)

FREQS = np.fft.rfftfreq(N_PERSEG, 1 / FS)
BAND = (FREQS >= F_MIN) & (FREQS <= F_MAX)
FREQS_B = FREQS[BAND]  # frequencies inside the range

BANDS = [(3.5, 4.5), (4.6, 5.8)]  # search window of each peak
MODE_COLS = [f"f{i + 1}" for i in range(len(BANDS))]


def check_data_dir():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found: replace DATA_DIR in config.py with the path to your data-z24 folder"
        )
