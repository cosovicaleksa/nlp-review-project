from pathlib import Path
import pickle
import joblib

_CACHE = {}

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def load_pickle_cached(path: Path):
    key = f"pickle::{path.resolve()}"
    if key not in _CACHE:
        _CACHE[key] = load_pickle(path)
    return _CACHE[key]

def load_joblib(path: Path):
    return joblib.load(path)

def load_joblib_cached(path: Path):
    key = f"joblib::{path.resolve()}"
    if key not in _CACHE:
        _CACHE[key] = load_joblib(path)
    return _CACHE[key]

    
    