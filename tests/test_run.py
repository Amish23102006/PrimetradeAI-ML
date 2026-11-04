"""Unit and end-to-end tests for the rolling-mean signal pipeline (run.py)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_config(path: Path, **overrides) -> Path:
    cfg = {"seed": 42, "window": 3, "version": "test"}
    cfg.update(overrides)
    lines = [f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}" for k, v in cfg.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


def run_cli(tmp_path: Path, input_path: Path, config_path: Path):
    out = tmp_path / "metrics.json"
    log = tmp_path / "run.log"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run.py"),
         "--input", str(input_path), "--config", str(config_path),
         "--output", str(out), "--log-file", str(log)],
        capture_output=True, text=True,
    )
    return proc, json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_load_config_valid(tmp_path):
    cfg = run.load_config(str(write_config(tmp_path / "c.yaml")))
    assert cfg == {"seed": 42, "window": 3, "version": "test"}


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.load_config(str(tmp_path / "missing.yaml"))


def test_load_config_missing_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 1\nwindow: 3\n")
    with pytest.raises(ValueError, match="missing required keys"):
        run.load_config(str(path))


@pytest.mark.parametrize("window", [0, -2])
def test_load_config_rejects_non_positive_window(tmp_path, window):
    with pytest.raises(ValueError, match="window"):
        run.load_config(str(write_config(tmp_path / "c.yaml", window=window)))


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        run.load_config(str(path))


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def test_load_dataset_missing_close_column(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("open,high\n1,2\n")
    with pytest.raises(ValueError, match="'close' not found"):
        run.load_dataset(str(path))


def test_load_dataset_empty(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("close\n")
    with pytest.raises(ValueError, match="empty"):
        run.load_dataset(str(path))


def test_load_dataset_drops_non_numeric_close(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("close\n1\nabc\n3\n")
    df = run.load_dataset(str(path))
    assert df["close"].tolist() == [1.0, 3.0]


def test_load_dataset_all_invalid_close(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("close\nx\ny\n")
    with pytest.raises(ValueError, match="no valid numeric"):
        run.load_dataset(str(path))


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def test_rolling_mean_warm_up_rows_are_nan():
    rm = run.compute_rolling_mean(pd.Series([1.0, 2.0, 3.0, 4.0]), window=3)
    assert rm.isna().tolist() == [True, True, False, False]
    assert rm.iloc[2] == pytest.approx(2.0)
    assert rm.iloc[3] == pytest.approx(3.0)


def test_signal_values_and_exclusion():
    close = pd.Series([1.0, 2.0, 3.0, 1.0])
    rm = run.compute_rolling_mean(close, window=3)
    signal = run.compute_signal(close, rm)
    assert np.isnan(signal.iloc[0]) and np.isnan(signal.iloc[1])
    assert signal.iloc[2] == 1  # 3 > mean(1, 2, 3) = 2
    assert signal.iloc[3] == 0  # 1 < mean(2, 3, 1) = 2


# ---------------------------------------------------------------------------
# End to end (CLI)
# ---------------------------------------------------------------------------

def test_cli_success_and_reproducible(tmp_path):
    data = tmp_path / "d.csv"
    pd.DataFrame({"close": np.arange(1, 21, dtype=float)}).to_csv(data, index=False)
    config = write_config(tmp_path / "c.yaml", window=5)

    proc1, m1 = run_cli(tmp_path, data, config)
    proc2, m2 = run_cli(tmp_path, data, config)

    assert proc1.returncode == 0
    assert m1["status"] == "success"
    assert m1["rows_processed"] == 20 - (5 - 1)
    assert m1["value"] == 1.0  # strictly increasing series: close > rolling mean
    assert (m1["value"], m1["rows_processed"]) == (m2["value"], m2["rows_processed"])


def test_cli_error_writes_error_metrics(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("open,high\n1,2\n")
    config = write_config(tmp_path / "c.yaml")

    proc, metrics = run_cli(tmp_path, data, config)

    assert proc.returncode == 1
    assert metrics["status"] == "error"
    assert metrics["version"] == "test"
    assert "close" in metrics["error_message"]
