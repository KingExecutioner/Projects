# Credit Risk Pipeline Simulation — Batch Ingestion, ML & Observability

A Colab notebook that simulates a recurring batch ML pipeline on top of a credit-risk
dataset: synthetic batches land on each run, get validated for schema drift, checked
for statistical data drift, feature-engineered, periodically used to retrain a model,
and logged for full lineage in SQLite — with an inline HTML observability dashboard
at the end. The notebook also includes a self-authored report proposing how each
piece would be re-architected for production scale on GCP.

## Contents

- `credit_risk_pipeline_simulation.ipynb` — the notebook (simulation + production
  architecture proposal, generated interactively).

## 1. Current architecture (as implemented in the notebook)

Everything runs inside a single Colab process, orchestrated by one Python function,
`run_once()`, called once per simulated batch arrival. There is no distributed
execution, no service boundaries, and no persistence beyond a Google Drive-mounted
folder (`/content/drive/MyDrive/credit_risk_pipeline`).

```
run_once(run_number, store):
  1. New batch        — generate_batch() creates ~250 synthetic rows per run,
                         with schema drift injected at run 6 (column rename) and
                         run 9 (new column), and a distribution shift ("macro shock")
                         injected at run 9+.
  2. Schema check      — snapshot_schema() infers dtypes for the incoming batch;
                         diff_schema() compares against EXPECTED_SCHEMA and flags
                         added/removed/type-changed columns.
  3. Append lake        — the new batch is concatenated (pd.concat, sort=False) onto
                         the full cumulative lake and the ENTIRE file is rewritten to
                         credit_risk_lake.csv. Column union handles schema drift, but
                         there is no atomicity — a crash mid-write can corrupt the file.
  4. Drift check        — compute_feature_drift() compares the new batch against the
                         very first batch (batch_000.csv) using Population Stability
                         Index (PSI) for numeric features, KS-test p-values, and simple
                         categorical distribution deltas. Severity is bucketed into
                         ok / warn / alert against fixed thresholds.
  5. Feature eng.       — engineer_features() runs on the FULL cumulative lake every
                         run (not just the new batch): imputes missing income/dependents,
                         clips outlier ages, bins age into bands, and derives ratios.
  6. Train or reuse     — a LogisticRegression + ColumnTransformer pipeline is retrained
                         every 3rd run (RETRAIN_EVERY = 3) or if no model exists yet;
                         otherwise the existing model.joblib is reused unchanged.
                         Fairness metrics (accuracy by age band, demographic parity
                         difference) are computed via fairlearn whenever training occurs.
  7. Log metadata       — every step above writes to a local SQLite database
                         (metadata.db) for lineage and observability (see below).
  8. Dashboard          — render_html() renders a static, self-contained HTML/JS
                         dashboard (Chart.js via CDN) summarizing runs, schema events,
                         and drift/model/fairness metrics pulled from SQLite.
```

Steps 1–8 repeat on every simulated batch (`run_simulation(n_runs=...)`), continuing
from the last run number stored in SQLite rather than resetting — the "recurring job"
behavior is entirely handled by application logic, not a scheduler.

### SQLite metadata & lineage store (`metadata.db`)

`MetadataStore` (backed by SQLite) is the pipeline's system of record. Six tables,
written to on every run:

| Table | Tracks |
|---|---|
| `runs` | One row per run: run number, started/finished timestamps, batch ID, batch row count, cumulative lake size, status, notes. |
| `lineage_edges` | A directed graph of data flow per run: `source_system → raw:<batch_id> → lake:credit_risk_lake → features:engineered → model:<version or "reused"> → predictions:<batch_id>`. Each edge carries a node kind, row count, and an MD5 checksum of the data at that point. |
| `schema_events` | Added columns, removed columns, and type changes detected per run, plus a `has_drift` flag. |
| `drift_metrics` | Per-feature PSI, KS statistic, KS p-value, and severity (ok/warn/alert) for every run where a reference batch was available. |
| `model_metrics` | Whether a model was trained that run, its version, AUC, accuracy, F1, and training row count (or a "reused" placeholder row on non-training runs). |
| `fairness_metrics` | Per-sensitive-group accuracy (age ≥65, age <25) and demographic parity difference, logged whenever a model is trained. |

This gives the notebook full lineage traceability and is what powers the dashboard's
queries (e.g. "show every run where drift severity != ok").

### Key limitations of the current design

- **No atomicity** — the full CSV lake is rewritten on every batch; a failure mid-write
  loses or corrupts data.
- **No horizontal scale** — reading/rewriting the whole lake and running feature
  engineering over the full cumulative history every run does not scale with data volume.
- **No service boundaries** — ingestion, validation, storage, training, and serving are
  all one function call; a failure anywhere aborts the whole run with no partial retry.
- **No real serving path** — the model is only ever scored in-process; there's no online
  or batch inference endpoint.
- **No managed schema evolution** — schema drift is detected but handled ad hoc (column
  union via `pd.concat`), not governed by compatibility rules.

## 2. Proposed production architecture (GCP)

The notebook's own analysis (see the "Report" and "Task" sections) proposes replacing
the single-process simulation with a decoupled, orchestrated system:

- **Ingestion** — Cloud Pub/Sub (streaming) and Cloud Storage (batch landing zone),
  replacing manual CSV drops.
- **Schema management** — a Schema Registry validating and versioning schemas on
  ingestion (Avro/Protobuf), enforcing backward/forward/full compatibility rules instead
  of ad hoc column unioning.
- **Curated data lake** — Delta Lake on GCS, giving ACID transactions, efficient
  appends, and native schema evolution in place of the CSV rewrite.
- **Feature store** — Vertex AI Feature Store, so training and serving compute features
  identically instead of duplicating `engineer_features()` logic per consumer.
- **Model training** — Vertex AI Training/Pipelines for scalable, reproducible runs,
  replacing the in-notebook `LogisticRegression.fit()` call.
- **Model registry** — versioned model artifacts (Vertex AI Model Registry) in place of
  a single mutable `model.joblib`.
- **Serving** — Vertex AI Endpoints for low-latency online inference, plus Vertex AI
  Batch Prediction for periodic offline scoring — neither of which the current
  simulation has at all.
- **Observability & monitoring** — Cloud Logging/Monitoring plus drift, fairness, and
  model-performance dashboards, generalizing the SQLite lineage store and end-of-run
  HTML dashboard into an always-on system with alerting.
- **Orchestration** — Vertex AI Pipelines or Cloud Composer (managed Airflow) as a DAG
  scheduler with per-step retries and recovery, replacing the linear `run_once()` /
  `run_simulation()` functions.

Full component-by-component rationale, including compatibility-mode definitions and
DAG stage breakdowns, is in the notebook's markdown cells.

## How to run

1. Open the notebook in Google Colab.
2. Run all cells top to bottom. The first run should call `run_simulation(n_runs=..., reset=True)`.
3. On subsequent sessions, call `run_simulation(n_runs=..., reset=False)` to continue
   the batch history from where it left off (state persists in the mounted Drive folder).
