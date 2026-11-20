from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from project291d.data.config import (
    BANDS,
    DAMAGE_START,
    DATA_DIR,
    DATES,
    EMS_CHANNELS,
    MODE_COLS,
    OUTPUT_DIR,
    check_data_dir,
)
from project291d.data.load_data import get_signatures, normalised_psd, peak_frequencies

EMS_PSD_CACHE = OUTPUT_DIR / "ems_psd.pkl"


@dataclass
class Environment:
    """What is needed to simulate observations"""
    clean: dict  # scenario -> frequencies without weather
    env_values: np.ndarray  # weather of each healthy hour
    env_effect: np.ndarray  # effect of that weather on the frequencies
    noise_std: np.ndarray  # sensor noise of each frequency


def get_environment() -> Environment:
    env_all, env_healthy = _load_environment()
    ems_feat = _get_ems_features()
    model, noise_std = _fit_environment_model(ems_feat, env_healthy)
    return Environment(
        clean=get_clean_pdt(model, env_all, env_healthy),
        env_values=env_healthy.to_numpy(),
        env_effect=model.predict(env_healthy),
        noise_std=noise_std,
    )


def get_clean_pdt(model, env_all: pd.DataFrame, env_healthy: pd.DataFrame) -> dict:
    """PDT signatures without the weather of their test day"""
    effect = {}
    for scenario, date in DATES.items():
        day = env_all[env_all.index.str.startswith(date)]  # weather of the test day
        if len(day) == 0:
            print(f"No weather data on {date}: using the healthy year")
            day = env_healthy
        effect[scenario] = model.predict(day).mean(axis=0)  # mean effect over the day
    effect = pd.DataFrame(effect, index=MODE_COLS).T

    clean = get_signatures() - effect  # remove the weather effect
    return {s: clean.loc[s].to_numpy() for s in clean.index}


def _load_environment() -> tuple:
    """Weather per hour: (whole period, healthy year)"""
    check_data_dir()

    # one row per hour, key yyyy/mm/dd/hh
    raw = pd.read_pickle(DATA_DIR / "Z24ems/env_data.pkl", compression="gzip").reset_index()
    raw["hour"] = raw["timestamp"].str[:13]
    env = raw.drop(columns=["timestamp", "row"]).set_index("hour")
    env = env[[c for c in env if not c.startswith("extra_")]]  # drop unused columns

    # make duplicate sensor names unique
    names = pd.Series(env.columns)
    names = names + np.where(names.duplicated(), "_" + names.groupby(names).cumcount().astype(str), "")
    env.columns = names

    env = env.where((env < 3000) & (env > -200))  # broken-sensor codes -> NaN
    env = env.groupby(level=0).mean()  # mean per hour

    # keep reliable columns, fill gaps with the median
    cols = [c for c in env if env[c].isna().mean() < 0.2 and env[c].std() > 1e-6]
    env_all = env[cols].fillna(env[cols].median())
    env_healthy = env_all[env_all.index < DAMAGE_START]  # before the damage tests
    return env_all, env_healthy


def _get_ems_features() -> pd.DataFrame:
    """Peak frequencies of every healthy EMS hour"""
    psd = _ems_mean_psd()
    features = pd.DataFrame(peak_frequencies(psd.to_numpy(), BANDS), index=psd.index, columns=MODE_COLS)
    features = features[psd.notna().all(axis=1)]  # drop hours without signal

    # drop outliers (sensor failures)
    deviation = (features - features.median()).abs()
    mad = deviation.median() * 1.4826
    return features[(deviation < 5 * mad).all(axis=1)]


def _fit_environment_model(ems_feat: pd.DataFrame, env_healthy: pd.DataFrame) -> tuple:
    """Weather -> frequencies. Returns (model, sensor noise std)"""
    # align features and weather by hour
    data = ems_feat.join(env_healthy, how="inner").sort_index()
    X, Y = data[env_healthy.columns], data[MODE_COLS]
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=0)

    # out-of-fold predictions, blocks of ~100 hours
    blocks = np.arange(len(data)) // 100
    cv_pred = cross_val_predict(model, X, Y, groups=blocks, cv=GroupKFold(5))
    noise_std = (Y.to_numpy() - cv_pred).std(axis=0)  # unexplained part = sensor noise

    return model.fit(X, Y), noise_std  # final fit on the whole year


def _ems_mean_psd() -> pd.DataFrame:
    """Normalised PSD per healthy hour, averaged over the 8 EMS sensors"""
    if EMS_PSD_CACHE.exists():
        return pd.read_pickle(EMS_PSD_CACHE)

    check_data_dir()
    frames = []
    for ch in EMS_CHANNELS:
        df = pd.read_pickle(DATA_DIR / f"Z24ems/data_channel_{ch}.pkl", compression="gzip")
        hours = [c for c in df.columns if c < DAMAGE_START]  # healthy hours only
        for i in range(0, len(hours), 250):  # chunks save memory
            chunk = hours[i:i + 250]
            psd = normalised_psd(df[chunk].to_numpy())
            frames.append(pd.DataFrame(psd.T.astype(np.float32), index=[h[:13] for h in chunk]))
        del df
        print(f"channel {ch} done")

    psd = pd.concat(frames).groupby(level=0).mean()  # mean over sensors
    EMS_PSD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    psd.to_pickle(EMS_PSD_CACHE)  # cache: slow to compute
    return psd


if __name__ == "__main__":
    env = get_environment()
    print(pd.DataFrame(env.clean, index=MODE_COLS).T)
    print("noise std (Hz):", env.noise_std.round(3))
