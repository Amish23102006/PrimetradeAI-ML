"""Unit and end-to-end tests for run.py. Run with: pytest -v"""
import json
import logging
import sys

import pandas as pd
import pytest

import run

GOOD_CONFIG = "seed: 42\nwindow: 3\nversion: v1\n"
# close = 1,2,3,2,5 with window 3 -> rolling mean: NaN, NaN, 2.0, 2.333, 3.333
# signal on the 3 valid rows: 1, 0, 1 -> signal_rate = 2/3
GOOD_CSV = "close\n1\n2\n3\n2\n5\n"


@pytest.fixture(autouse=True)
def reset_logger():
    """run.setup_logging() adds handlers on every call; clean them up between tests."""
    yield
    logger = logging.getLogger("mlops")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def run_job(workdir, monkeypatch, csv=GOOD_CSV, config=GOOD_CONFIG):
    """Write input files into workdir, run run.main(), return (exit_code, metrics_dict)."""
    workdir.mkdir(exist_ok=True)
    inp, cfg = workdir / "data.csv", workdir / "config.yaml"
    out, log = workdir / "metrics.json", workdir / "run.log"
    if csv is not None:
        inp.write_text(csv, encoding="utf-8")
    if config is not None:
        cfg.write_text(config, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--input", str(inp), "--config", str(cfg),
        "--output", str(out), "--log-file", str(log),
    ])
    code = run.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- processing
def test_rolling_mean_has_warmup_nans():
    result = run.compute_rolling_mean(pd.Series([1, 2, 3, 4]), window=2)
    assert result.isna().tolist() == [True, False, False, False]
    assert result.dropna().tolist() == [1.5, 2.5, 3.5]


def test_signal_excludes_warmup_rows():
    close = pd.Series([1, 2, 3, 2, 5])
    signal = run.compute_signal(close, run.compute_rolling_mean(close, window=3))
    assert signal.isna().tolist() == [True, True, False, False, False]
    assert signal.dropna().tolist() == [1.0, 0.0, 1.0]


# -------------------------------------------------------------------- config
def test_load_config_valid(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(GOOD_CONFIG)
    assert run.load_config(str(path)) == {"seed": 42, "window": 3, "version": "v1"}


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.load_config(str(tmp_path / "nope.yaml"))


def test_load_config_missing_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\n")
    with pytest.raises(ValueError, match="missing required keys"):
        run.load_config(str(path))


@pytest.mark.parametrize("text, match", [
    ("seed: 42\nwindow: 0\nversion: v1\n", "window"),
    ("seed: 42\nwindow: abc\nversion: v1\n", "window"),
    ("seed: abc\nwindow: 3\nversion: v1\n", "seed"),
    ("seed: 42\nwindow: 3\nversion: ''\n", "version"),
    ("- just\n- a list\n", "mapping"),
])
def test_load_config_invalid_values(tmp_path, text, match):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        run.load_config(str(path))


# ------------------------------------------------------------------- dataset
def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run.load_dataset(str(tmp_path / "nope.csv"))


def test_load_dataset_empty_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("close\n")
    with pytest.raises(ValueError, match="empty"):
        run.load_dataset(str(path))


def test_load_dataset_completely_empty_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("")
    with pytest.raises(ValueError):
        run.load_dataset(str(path))


def test_load_dataset_missing_close_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("open,high\n1,2\n")
    with pytest.raises(ValueError, match="Required column 'close'"):
        run.load_dataset(str(path))


def test_load_dataset_all_non_numeric(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("close\nabc\nxyz\n")
    with pytest.raises(ValueError, match="no valid numeric"):
        run.load_dataset(str(path))


def test_load_dataset_drops_non_numeric_rows(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("close\n1\nabc\n3\n")
    df = run.load_dataset(str(path))
    assert df["close"].tolist() == [1.0, 3.0]


# --------------------------------------------------------- end-to-end (main)
def test_main_success(tmp_path, monkeypatch, capsys):
    code, metrics = run_job(tmp_path, monkeypatch)
    assert code == 0
    assert metrics["status"] == "success"
    assert metrics["version"] == "v1"
    assert metrics["seed"] == 42
    assert metrics["metric"] == "signal_rate"
    assert metrics["rows_processed"] == 3
    assert metrics["value"] == pytest.approx(0.6667)
    assert metrics["latency_ms"] >= 0
    assert (tmp_path / "run.log").exists()
    # the same JSON is printed to stdout
    assert json.loads(capsys.readouterr().out) == metrics


def test_main_is_reproducible(tmp_path, monkeypatch):
    _, first = run_job(tmp_path / "a", monkeypatch)
    _, second = run_job(tmp_path / "b", monkeypatch)
    assert first["value"] == second["value"]
    assert first["rows_processed"] == second["rows_processed"]


def test_main_missing_input_file(tmp_path, monkeypatch):
    code, metrics = run_job(tmp_path, monkeypatch, csv=None)
    assert code == 1
    assert metrics["status"] == "error"
    assert metrics["version"] == "v1"
    assert "not found" in metrics["error_message"]


def test_main_missing_close_column(tmp_path, monkeypatch):
    code, metrics = run_job(tmp_path, monkeypatch, csv="open,high\n1,2\n")
    assert code == 1
    assert metrics["status"] == "error"
    assert "close" in metrics["error_message"]


def test_main_bad_config_writes_error_metrics(tmp_path, monkeypatch):
    code, metrics = run_job(tmp_path, monkeypatch, config="seed: 42\n")
    assert code == 1
    assert metrics["status"] == "error"
    assert metrics["version"] == "unknown"
