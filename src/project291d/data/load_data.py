from itertools import product

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

from project291d.data.config import (
    BAND,
    BANDS,
    DATA_DIR,
    FREQS_B,
    FS,
    MODE_COLS,
    N_PERSEG,
    OUTPUT_DIR,
    SCENARIOS,
    check_data_dir,
)

CACHE_FILE = OUTPUT_DIR / "record_features.pkl"


def get_signatures(use_cache: bool = True) -> pd.DataFrame:
    """One feature vector per scenario (mean of its 9 setups)"""
    return get_record_features(use_cache).groupby(level=0).mean()


def get_record_features(use_cache: bool = True) -> pd.DataFrame:
    """Peak frequencies of every AVT record (rows: scenario, setup)"""
    if use_cache and CACHE_FILE.exists():
        return pd.read_pickle(CACHE_FILE)

    # PSD of each record, then its peak frequencies
    psd = _record_psd()
    features = pd.DataFrame(peak_frequencies(psd.to_numpy(), BANDS), index=psd.index, columns=MODE_COLS)

    # cache so .mat files are read once
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(CACHE_FILE)
    return features


def load_data(scenario: int = 1, setup: int = 1, vt_type: str = "avt") -> pd.DataFrame:
    """Load one record. FVT setups are unified: FVT setup 1 calls file 09"""
    check_data_dir()

    # build the file name
    file_path = f"pdt/{scenario:02d}/{vt_type}/{scenario:02d}setup"
    if vt_type.lower() == "avt":
        file_path += f"{setup:02d}.mat"
    elif vt_type.lower() == "fvt":
        file_path += f"{10 - setup:02d}.mat"
    else:
        raise ValueError(f"Unknown vibration test: {vt_type}")

    # one column per sensor
    data = loadmat(DATA_DIR / file_path)
    df = pd.DataFrame(data["data"], columns=data["labelshulp"])
    df.columns = df.columns.str.strip()
    return df


def normalised_psd(x: np.ndarray) -> np.ndarray:
    """PSD of each column, divided by its total power (NaN if no signal)"""
    # Welch PSD of every sensor
    _, psd = welch(x, fs=FS, nperseg=N_PERSEG, detrend="linear", axis=0)
    psd = psd[BAND]  # keep the range of interest

    # normalise by total power
    with np.errstate(invalid="ignore", divide="ignore"):
        psd = psd / psd.sum(axis=0)
    psd[:, ~np.isfinite(psd).all(axis=0)] = np.nan  # flag broken sensors
    return psd


def peak_frequencies(psd: np.ndarray, bands: list) -> np.ndarray:
    """Frequency of the PSD maximum inside each band (n_rows, n_bands)"""
    out = []
    for lo, hi in bands:
        in_band = (FREQS_B >= lo) & (FREQS_B <= hi)  # window of this peak
        out.append(FREQS_B[in_band][np.argmax(psd[:, in_band], axis=1)])
    return np.column_stack(out)


def _record_psd() -> pd.DataFrame:
    """PSD of every AVT record, averaged over its sensors"""
    check_data_dir()
    pdt_psd = {}
    for scenario, setup in product(SCENARIOS, range(1, 10)):
        psd = normalised_psd(load_data(scenario, setup, "avt").to_numpy())
        pdt_psd[(scenario, setup)] = np.nanmean(psd, axis=1)  # mean over sensors
    return pd.DataFrame(pdt_psd).T


if __name__ == "__main__":
    get_record_features(use_cache=False)
    print(get_signatures())
