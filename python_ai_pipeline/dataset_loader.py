import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


CLASS_NAME_TO_ID = {
    "normal": 0,
    "imbalance": 1,
    "bearing_fault": 2,
    "bearing-fault": 2,
    "bearing fault": 2,
    "misalignment": 3,
}

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


def _detect_sep(csv_path: str) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        head = f.readline()
    # Our data appears to use ';' in provided example; fallback to comma
    if ";" in head:
        return ";"
    return ","


def _read_class_csv(csv_path: str) -> pd.DataFrame:
    sep = _detect_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    # Normalize expected columns
    # Accept case-insensitive and common variants
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("x_axis_mg", "x", "axis_x", "x_axis"):
            col_map[c] = "X_axis_mg"
        elif lc in ("y_axis_mg", "y", "axis_y", "y_axis"):
            col_map[c] = "Y_axis_mg"
        elif lc in ("z_axis_mg", "z", "axis_z", "z_axis"):
            col_map[c] = "Z_axis_mg"
        elif lc in ("timestamp", "time", "t"):
            col_map[c] = "timestamp"
        elif lc in ("fault_type", "label", "class"):
            col_map[c] = "fault_type"
        elif lc in ("sample_id", "index", "idx"):
            col_map[c] = "sample_id"
    if col_map:
        df = df.rename(columns=col_map)
    required = ["X_axis_mg", "Y_axis_mg", "Z_axis_mg"]
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Missing column '{req}' in {csv_path}")
    # Ensure timestamp exists; if not, synthesize based on index
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df), dtype=float)
    return df


def _read_metadata(meta_path: str) -> Dict:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_dataset(base_dir: str = "collected_data") -> List[Dict]:
    """
    Discover class folders and load CSV + metadata for each class.
    Returns a list of dicts with keys: class_name, class_id, df, meta.
    """
    results: List[Dict] = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    for entry in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(class_dir):
            continue
        key = entry.strip().lower()
        if key not in CLASS_NAME_TO_ID:
            # tolerate 'bearing_fault' directory name
            key = key.replace("_", "-")
        if key not in CLASS_NAME_TO_ID:
            continue

        class_id = CLASS_NAME_TO_ID[key]
        class_name = entry
        # Find CSV
        csv_files = [f for f in os.listdir(class_dir) if f.lower().endswith(".csv")]
        if not csv_files:
            continue
        # Prefer file containing 'vibration' in name
        csv_files.sort(key=lambda n: ("vibration" not in n.lower(), n))
        csv_path = os.path.join(class_dir, csv_files[0])

        # Metadata
        meta_files = [f for f in os.listdir(class_dir) if f.lower().endswith(".json")]
        meta_path = os.path.join(class_dir, meta_files[0]) if meta_files else ""
        meta = _read_metadata(meta_path) if meta_path else {}

        df = _read_class_csv(csv_path)
        results.append({
            "class_name": class_name,
            "class_id": class_id,
            "df": df,
            "meta": meta,
            "csv_path": csv_path,
            "meta_path": meta_path,
        })
    if not results:
        raise RuntimeError("No class CSVs found in collected_data/*/")
    return results


def infer_sample_rate(df: pd.DataFrame, declared_hz: Optional[float]) -> float:
    # Prefer inferring from timestamps if they look sane
    t = df["timestamp"].to_numpy()
    if len(t) >= 3:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt)]
        if dt.size:
            median_dt = float(np.median(dt))
            if median_dt > 0:
                inferred = 1.0 / median_dt
                # If declared differs wildly, trust data
                if not declared_hz or abs(inferred - declared_hz) / max(declared_hz, 1e-6) > 0.2:
                    return inferred
    # Fallback to declared if provided
    if declared_hz and declared_hz > 0:
        return float(declared_hz)
    # Final fallback
    return 1000.0


def make_windows(df: pd.DataFrame,
                 sample_rate_hz: float,
                 window_seconds: float = 2.0,
                 step_seconds: float = 0.5) -> np.ndarray:
    x = df["X_axis_mg"].to_numpy(dtype=np.float32)
    y = df["Y_axis_mg"].to_numpy(dtype=np.float32)
    z = df["Z_axis_mg"].to_numpy(dtype=np.float32)
    n = len(x)
    requested_win = int(round(window_seconds * sample_rate_hz))
    step = int(round(step_seconds * sample_rate_hz))
    if requested_win <= 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    # If window is longer than the sequence, adapt to available length
    if n < requested_win:
        win_len = max(32, int(max(8, 0.8 * n)))
        step = max(1, min(step, win_len // 2))
    else:
        win_len = requested_win
    if step <= 0:
        step = max(1, win_len // 4)
    windows: List[np.ndarray] = []
    for start in range(0, n - win_len + 1, step):
        sl = slice(start, start + win_len)
        win = np.stack([x[sl], y[sl], z[sl]], axis=-1)  # (win_len, 3)
        windows.append(win)
    return np.stack(windows, axis=0) if windows else np.zeros((0, 0, 0), dtype=np.float32)


def build_dataset(base_dir: str = "collected_data",
                  window_seconds: float = 2.0,
                  step_seconds: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Returns: X (N, T, 3), y (N,), inferred_sample_rate, info dict
    """
    discovered = discover_dataset(base_dir)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    per_class_counts: Dict[int, int] = {}
    sample_rates: List[float] = []

    for item in discovered:
        df = item["df"]
        meta = item.get("meta", {})
        di = meta.get("dataset_info", {})
        declared_sr = di.get("sample_rate_hz")
        sr = infer_sample_rate(df, declared_sr)
        sample_rates.append(sr)

        windows = make_windows(df, sr, window_seconds, step_seconds)
        if windows.size == 0:
            continue
        X_list.append(windows)
        y_list.append(np.full((windows.shape[0],), item["class_id"], dtype=np.int64))
        per_class_counts[item["class_id"]] = per_class_counts.get(item["class_id"], 0) + windows.shape[0]

    if not X_list:
        raise RuntimeError("No windows could be constructed from the dataset")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # Use median of inferred sample rates
    inferred_sr = float(np.median(np.array(sample_rates))) if sample_rates else 1000.0

    info = {
        "per_class_counts": per_class_counts,
        "class_map": CLASS_NAME_TO_ID,
        "id_to_class": ID_TO_CLASS_NAME,
        "inferred_sample_rate": inferred_sr,
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
    }
    return X, y, inferred_sr, info


