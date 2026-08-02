"""Shared utility for converting numpy types to native Python for JSON serialization."""
import numpy as np


def pyval(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, dict):
        return {k: pyval(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [pyval(x) for x in v]
    return v
