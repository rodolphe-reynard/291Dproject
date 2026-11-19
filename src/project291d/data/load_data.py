from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

# TODO this is my personal path, you need to change this to your path
DATA_DIR = Path("C:/Users/Home/Code/291Dproject/data-z24")
CACHE_FILE = Path("outputs/record_features.pkl")

# The 17 PDT scenarios
SCENARIOS = list(range(1, 18))

FS = 100.0  # sampling frequency (Hz), it's the same for PDT and EMS
N_PERSEG = 4096  # Welch segment: frequency resolution of 0.024 Hz
F_MIN, F_MAX = (1.0, 25.0)  # frequency range of interest (Hz), the anti-aliasing cutoff is at 30 Hz

FREQS = np.fft.rfftfreq(N_PERSEG, 1 / FS)
BAND = (FREQS >= F_MIN) & (FREQS <= F_MAX)
FREQS_B = FREQS[BAND]
BANDS = [(3.5, 4.5), (4.6, 5.8)]  # range where the peak frequencies sit
MODE_COLS = [f"f{i + 1}" for i in range(len(BANDS))]


def get_signatures(use_cache: bool = True) -> pd.DataFrame:
    """One feature vector per scenario: the mean of its 9 setups"""
    return get_record_features(use_cache).groupby(level=0).mean()


def get_record_features(use_cache: bool = True) -> pd.DataFrame:
    """Peak frequencies of every AVT record: 17 scenarios x 9 setups (rows: (scenario, setup)).
    The result is cached, so the .mat files are only read once."""
    if use_cache and CACHE_FILE.exists():
        return pd.read_pickle(CACHE_FILE)

    psd = _record_psd()
    features = pd.DataFrame(peak_frequencies(psd.to_numpy(), BANDS), index=psd.index, columns=MODE_COLS)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(CACHE_FILE)
    return features


def load_data(scenario: int = 1, setup: int = 1, vt_type: str = "avt") -> pd.DataFrame:
    """
    Load the data based on the scenario, setup and type of vibration test.\\
    The **setups are unified**, meaning that FVT setup 1 will call setup 9.
    """
    file_path = f"pdt/{scenario:02d}/{vt_type}/{scenario:02d}setup"

    if vt_type.lower() == "avt":
        file_path += f"{setup:02d}.mat"
    elif vt_type.lower() == "fvt":
        file_path += f"{10 - setup:02d}.mat"  # unify the setups
    else:
        raise ValueError(f"Unkown vibration test: {vt_type}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found: replace DATA_DIR at the top of load_data.py with the path to your data-z24 folder"
        )
    data = loadmat(DATA_DIR / file_path)
    df = pd.DataFrame(data["data"], columns=data["labelshulp"])
    df.columns = df.columns.str.strip()

    return df


def normalised_psd(x: np.ndarray) -> np.ndarray:
    """PSD of each column of x (n_samples, n_columns) in [F_MIN, F_MAX], each divided by its own total power.
    Columns with NaN or no signal give NaN. Public: the EMS code must use this same function."""
    _, psd = welch(x, fs=FS, nperseg=N_PERSEG, detrend="linear", axis=0)
    psd = psd[BAND]
    with np.errstate(invalid="ignore", divide="ignore"):
        psd = psd / psd.sum(axis=0)
    psd[:, ~np.isfinite(psd).all(axis=0)] = np.nan
    return psd


def peak_frequencies(psd: np.ndarray, bands: list) -> np.ndarray:
    """Frequency of the maximum of each PSD (rows of psd) inside each band -> (n_rows, n_bands).
    Public: the EMS code must use this same function."""
    out = []
    for lo, hi in bands:
        m = (FREQS_B >= lo) & (FREQS_B <= hi)
        out.append(FREQS_B[m][np.argmax(psd[:, m], axis=1)])
    return np.column_stack(out)


def _record_psd() -> pd.DataFrame:
    """PSD of every AVT record (mean over its sensors): 17 scenarios x 9 setups"""
    pdt_psd = {}
    for scenario, setup in product(SCENARIOS, range(1, 10)):
        psd = normalised_psd(load_data(scenario, setup, "avt").to_numpy())
        pdt_psd[(scenario, setup)] = np.nanmean(psd, axis=1)
    return pd.DataFrame(pdt_psd).T


if __name__ == "__main__":
    features = get_record_features(use_cache=False)
    print(get_signatures())
