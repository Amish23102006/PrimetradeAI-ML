# MLOps Batch Job: Rolling-Mean Signal Pipeline

A minimal MLOps-style batch job that demonstrates **reproducibility**, **observability**, and **deployment readiness** through a Dockerised, one-command pipeline.

**Tech stack:** Python · pandas / NumPy · YAML config · Docker

---

## What it does

| Step | Description |
|------|-------------|
| 1 | Load and validate `config.yaml` (seed, window, version) |
| 2 | Load and validate `data.csv`: checks for missing file, bad CSV, empty file, and missing `close` column |
| 3 | Compute a rolling mean on `close` with a configurable `window` |
| 4 | Generate a binary signal: `1` if `close > rolling_mean`, else `0` |
| 5 | Write a structured `metrics.json` and a detailed `run.log` |

The first `window - 1` rows produce `NaN` rolling-mean values. They are **excluded** from signal computation and metric counts, consistently and by design.

---

## Project structure

```
.
├── run.py            # Main pipeline
├── config.yaml       # Seed, window, version config
├── data.csv          # 10,000-row OHLCV dataset
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build spec
├── metrics.json      # Sample successful output
├── run.log           # Sample log output
└── README.md
```

---

## Local run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

The final metrics JSON is printed to **stdout**; structured logs go to `run.log`.

---

## Docker build and run

```bash
# Build
docker build -t mlops-task .

# Run
docker run --rm mlops-task
```

- Exit code `0` means success
- A non-zero exit code means failure (details in `metrics.json` and stdout)

Copy outputs out of the container (optional):
```bash
docker run --rm -v "$(pwd)/out":/app/out mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   out/metrics.json \
    --log-file out/run.log
```

---

## Config reference (`config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `seed` | int | NumPy random seed for reproducibility |
| `window` | int ≥ 1 | Rolling-mean window size |
| `version` | str | Pipeline version tag written to metrics |

---

## Input requirements (`data.csv`)

- Must be a valid, non-empty CSV file.
- Must contain a numeric `close` column.
- Rows where `close` is non-numeric are dropped with a warning.
- Other columns (`open`, `high`, `low`, `volume`) are ignored by the pipeline.

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.5118,
  "latency_ms": 41,
  "seed": 42,
  "status": "success"
}
```
`rows_processed` is the total row count minus the `window - 1` warm-up rows.

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

`metrics.json` is **always written**, even on failure, so downstream monitors always have a machine-readable status.

---

## Reproducibility

Running the pipeline repeatedly with the same `config.yaml` and `data.csv` produces **identical** `value` and `rows_processed` outputs. Only `latency_ms` varies with hardware load.

---

## Validation errors handled

| Case | Behaviour |
|------|-----------|
| Missing input file | Error metrics written, exit 1 |
| Non-parseable CSV | Error metrics written, exit 1 |
| Empty CSV | Error metrics written, exit 1 |
| Missing `close` column | Error metrics written, exit 1 |
| Missing config keys | Error metrics written, exit 1 |
| Non-numeric `close` rows | Warning logged, rows dropped, pipeline continues |

---

**Author:** Amish Chaturvedi · [GitHub](https://github.com/Amish23102006) · [LinkedIn](https://linkedin.com/in/amish-chaturvedi-756606333)
