from pathlib import Path
import pickle

_CACHE = {}

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def load_pickle_cached(path: Path):
    key = str(path.resolve())
    if key not in _CACHE:
        _CACHE[key] = load_pickle(path)
    return _CACHE[key]

    
    