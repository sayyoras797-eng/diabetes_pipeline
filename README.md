# Early Prediction of Diabetes Risk — Data Pipeline

## Architecture

```
Dagster schedule (hourly, 09:00–17:00, Mon–Fri, Asia/Tashkent)
    └─► generate_patients   (patient data generator)
            ├─► write_to_postgres   ─► PostgreSQL (patient_records table)
            │                              └─► FastAPI (REST API)
            │                                      ├─► Streamlit Dashboard
            │                                      │      (PDF report export,
            │                                      │       ML risk calculator)
            │                                      └─► /predict (ML model)
            └─► send_high_risk_alert ─► email (optional, via SMTP)

ml-trainer (one-off job, optional "tools" profile)
    └─► loads patient_records ─► trains models ─► optionally logs to MLflow
            └─► saves best model ─► shared volume ─► used by FastAPI /predict

nginx reverse proxy (port 80)
    ├─► /            → Streamlit Dashboard
    └─► /dagster/    → Dagster UI (HTTP Basic Auth) — pipeline run monitoring
```

All services run in Docker containers. Only `nginx` is exposed to the
host (port 80) — Postgres, Dagster, FastAPI and Streamlit are only
reachable from inside the Docker network.

MLflow and `ml-trainer` are behind the `tools` profile and **do not
start by default** — they're only needed when retraining the model, not
for serving `/predict`. This keeps the default deployment lightweight
(~650MB RAM instead of ~1.2GB), which matters on small free-tier VPS
instances.

Dagster's code (jobs, ops, schedule) runs in its own `dagster-user-code`
gRPC server container; `dagster-webserver` and `dagster-daemon` both
connect to it. This is the standard production pattern and avoids each
process spawning its own short-lived code server.

## Quick start

```bash
# 1. Enter project
cd diabetes_pipeline

# 2. Build and start everything
docker compose up -d --build

# 3. Open in browser:
#    Streamlit dashboard → http://localhost            (login: see DASHBOARD_PASSWORD)
#    Dagster UI          → http://localhost/dagster/   (Basic Auth)
#
#    Basic Auth credentials for /dagster/ live in
#    nginx/.htpasswd (currently admin / 0334 — regenerate with:
#      openssl passwd -apr1 'your-new-password')
```

The Dagster schedule (`diabetes_hourly_schedule`) starts in a
**stopped** state by default. To activate it, open the Dagster UI →
Overview → Schedules → toggle `diabetes_hourly_schedule` on. You can
also trigger the `diabetes_hourly_pipeline` job manually from the
Dagster UI Launchpad to generate data immediately, without waiting for
the schedule.

## Production deployment (VPS, IP-based, no domain)

These steps deploy the whole stack to a Linux server (e.g. Ubuntu
22.04) reachable by IP only — no domain/DNS or HTTPS required.

### 1. Server requirements

- Ubuntu 22.04 (or any Linux with Docker support)
- At least 1 vCPU / 1 GB RAM for the default stack (Dagster + Postgres +
  FastAPI + Streamlit + nginx, ~650MB). The optional `tools` profile
  (MLflow + ml-trainer, for retraining only) needs ~1.5GB extra.
- Open inbound port **80** (and **22** for SSH)

### 2. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker compose version   # confirm the Compose plugin is available
```

### 3. Copy the project to the server

```bash
# from your local machine
scp -r diabetes_pipeline user@SERVER_IP:/opt/diabetes_pipeline
# or, if the project is in a git repo:
git clone <repo-url> /opt/diabetes_pipeline
```

### 4. Configure secrets

```bash
cd /opt/diabetes_pipeline
cp .env.example .env
nano .env   # set strong POSTGRES_PASSWORD and DASHBOARD_PASSWORD,
            # optionally fill in SMTP_* / ALERT_EMAIL_* for email alerts
```

Also regenerate the Basic Auth credentials for `/dagster/`
(default is `admin` / `0334`):

```bash
sudo apt-get install -y apache2-utils   # provides htpasswd
htpasswd -c nginx/.htpasswd admin       # prompts for a new password
```

### 5. Start the stack

```bash
docker compose up -d --build
```

This starts Postgres, the Dagster code server/webserver/daemon,
FastAPI, Streamlit and nginx. The `tunnel` (Cloudflare), `mlflow` and
`ml-trainer` services do **not** start automatically — they're behind
`profiles`.

### 6. Train the initial model

```bash
docker compose --profile tools run --rm ml-trainer
```

Without this step, `/predict` returns 503 until a model exists. This
also starts `mlflow` (its `depends_on`) for experiment tracking —
visit `http://SERVER_IP:5000`. Stop it afterwards with
`docker compose --profile tools stop mlflow` to free up RAM.

### 7. Open the firewall

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw enable
# Optional, only if you'll use the MLflow UI (tools profile):
# sudo ufw allow 5000/tcp
```

### 8. Access

- Dashboard: `http://SERVER_IP/`
- Dagster UI: `http://SERVER_IP/dagster/` (Basic Auth)
- MLflow UI (only while the `tools` profile is running): `http://SERVER_IP:5000`

Activate the hourly schedule as described above (Dagster UI →
Overview → Schedules).

### Notes

- Docker's `restart: unless-stopped` / `on-failure` policies bring
  services back up automatically after a server reboot (as long as the
  Docker daemon itself starts on boot, which `get.docker.com` enables
  by default).
- To upgrade later: `git pull` (or re-copy files), then
  `docker compose up -d --build`.
- HTTPS requires a domain pointed at the server (for Let's Encrypt /
  certbot). Without a domain, IP + HTTP is the practical option; avoid
  sending real patient data over this setup on an untrusted network.

## Model training

The ML model used by `/predict` is trained by a one-off container and
is **not** part of the default `docker compose up`:

```bash
docker compose --profile tools run --rm ml-trainer
```

This loads `patient_records` from PostgreSQL, trains Logistic
Regression, Random Forest, Gradient Boosting and SVM models, logs all
metrics/plots/models to MLflow (started automatically as a dependency,
`http://SERVER_IP:5000`), picks the best model by accuracy, and saves
it to a shared volume so FastAPI's `/predict` endpoint can serve it
immediately. Re-run this whenever you want to retrain on fresh data.

## Email alerts for high-risk patients

After each pipeline run, the `send_high_risk_alert` op checks the new
batch for patients with `diabetes_risk >= 0.8`. If any are found and
SMTP is configured, it sends a summary email. If `SMTP_HOST` is not
set, the op logs the finding and skips sending — no error.

To enable, set these environment variables (e.g. in a `.env` file next
to `docker-compose.yml`):

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server hostname (e.g. `smtp.gmail.com`) |
| `SMTP_PORT` | SMTP port (default `587`) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASSWORD` | SMTP password / app password |
| `ALERT_EMAIL_FROM` | Sender address (defaults to `SMTP_USER`) |
| `ALERT_EMAIL_TO` | Recipient address for alerts |

## Pipeline schedule

| Setting    | Value                                          |
|------------|------------------------------------------------|
| Job        | `diabetes_hourly_pipeline`                     |
| Schedule   | `diabetes_hourly_schedule` — `0 9-17 * * 1-5`  |
| Runs       | Every hour on the hour, Mon–Fri                |
| Window     | 09:00 – 17:00 (Asia/Tashkent)                   |
| Steps      | `generate_patients` → `write_to_postgres` → `send_high_risk_alert` → `log_summary` |
| Batch size | 50 patient records per run                      |
| Storage    | PostgreSQL — `patient_records` table (Dagster run/event/schedule storage also lives in the same Postgres instance) |

## API Endpoints

| Method | Path            | Description                        |
|--------|-----------------|-------------------------------------|
| GET    | /patients       | List records (limit, label filter) |
| GET    | /stats          | Aggregate statistics               |
| GET    | /health         | Health check                       |
| POST   | /predict        | Diabetes risk prediction from the trained ML model |

## Patient data schema

| Column          | Type    | Range / Values          |
|-----------------|---------|--------------------------|
| age             | int     | 20–75                    |
| gender          | str     | Male / Female            |
| bmi             | float   | 15.0–55.0                |
| glucose         | float   | 60.0–280.0 mg/dL         |
| blood_pressure  | float   | 40.0–130.0 mmHg          |
| hba1c           | float   | 4.0–14.0 %               |
| insulin         | float   | 0.0–500.0 μU/mL          |
| skin_thickness  | float   | 5.0–80.0 mm              |
| pregnancies     | int     | 0–10 (Female only)       |
| physical_activity | str   | Low / Medium / High      |
| smoking_status  | str     | Never / Former / Current |
| alcohol_consumption | str | None / Moderate / High   |
| family_history  | int     | 0 = no, 1 = yes (diabetes in family) |
| diabetes_risk   | float   | 0.0–1.0 (logistic score) |
| label           | int     | 0 = negative, 1 = positive |

## Dataset realism & methodology

The pipeline generates **synthetic** patient records (`generator/data_generator.py`)
instead of using a real clinical dataset. This is a deliberate choice,
not a shortcut:

- **Patient privacy**: real glucose/HbA1c/insulin records are personal
  health data (PHI). Using or redistributing them would require
  ethics approval and data-sharing agreements that are out of scope
  for a student project.
- **Reproducibility & volume**: the pipeline needs a continuously
  growing stream of records (hourly batches) to demonstrate
  orchestration, monitoring and model retraining — a fixed real
  dataset can't provide that.

To still make the data clinically credible, every feature is generated
from **published diagnostic ranges and known physiological
relationships**, not arbitrary random numbers:

| Feature | Modeled range | Clinical basis |
|---------|---------------|-----------------|
| Age | 20–75 | General adult screening population |
| BMI | ~28.6 ± 5.9 | WHO BMI categories (18.5 normal, 25–30 overweight, 30+ obese) |
| Glucose (fasting) | ~94.7 ± 20.0 mg/dL | ADA: <100 normal, 100–125 prediabetes, ≥126 diabetes |
| HbA1c | ~4.9 ± 0.6 % | ADA: <5.7% normal, 5.7–6.4% prediabetes, ≥6.5% diabetes |
| Blood pressure | ~73.4 ± 9.1 mmHg | Normal diastolic range (60–80 mmHg) |
| Insulin | ~51.5 ± 32.7 μU/mL | Normal fasting insulin (2–25 μU/mL) with insulin-resistance tail |
| Skin thickness | ~20.4 ± 6.4 mm | Triceps skinfold, correlates with body fat / BMI |
| Pregnancies | 0–10 (female only) | Standard field used in diabetes risk studies (e.g. Pima Indians Diabetes Dataset) |
| Physical activity | Low / Medium / High | WHO physical activity guidelines; inactivity is a major modifiable diabetes risk factor |
| Smoking status | Never / Former / Current | ADA: smoking increases insulin resistance and Type 2 diabetes risk |
| Alcohol consumption | None / Moderate / High | Heavy alcohol use is linked to impaired glucose tolerance |
| Family history | Yes / No | Genetic predisposition — one of the strongest known risk factors (ADA) |

**Modeled correlations** (mirroring real clinical relationships):
glucose rises with age and BMI; HbA1c tracks glucose (long-term
control); blood pressure and insulin resistance both rise with BMI;
skinfold thickness tracks body fat; low physical activity raises BMI
and insulin resistance. A small random "unmeasured factors" term
(other genetics, diet details) is added so the outcome isn't a perfect
deterministic function of the measured features — just like in real
data.

**Label / prevalence**: `diabetes_risk` is a logistic function of the
weighted clinical and lifestyle risk factors above (including family
history, smoking, alcohol use and physical activity). The intercept is
calibrated so that **~15% of records are labeled diabetic**, matching
real-world adult screening prevalence (IDF/WHO global estimates: ~10–15%).

With this design, a 10,000-record dataset (`data/patient_records.csv`,
refreshed from `patient_records` via `\copy ... TO STDOUT WITH CSV
HEADER`) has distributions and a class balance consistent with
published clinical references — while remaining fully synthetic and
privacy-safe.

## Model training (optional, standalone)

`ml/train_model.py` is a standalone script (not part of the Docker
stack) that trains and compares Logistic Regression, Random Forest,
Gradient Boosting and SVM models on the data in `patient_records`,
logging metrics, plots and models to an MLflow tracking server.

```bash
# Requires your own MLflow tracking server (set MLFLOW_TRACKING_URI),
# plus: pip install -r ml/requirements.txt
POSTGRES_HOST=localhost MLFLOW_TRACKING_URI=http://localhost:5000 \
  python ml/train_model.py --experiment "diploma-v1"
```

## Project structure

```
diabetes_pipeline/
├── docker-compose.yml
├── init.sql                       ← PostgreSQL schema (patient_records)
├── __init__.py
├── dagster_pipeline/
│   ├── Dockerfile
│   ├── dagster.yaml               ← Postgres-backed run/event/schedule storage
│   ├── workspace.yaml             ← points at dagster-user-code gRPC server
│   ├── requirements.txt
│   ├── __init__.py
│   └── jobs/
│       ├── __init__.py
│       └── diabetes_job.py        ← job + hourly schedule + email alert op
├── generator/
│   ├── __init__.py
│   └── data_generator.py          ← patient data generator
├── backend/
│   ├── main.py                    ← FastAPI endpoints (incl. /predict)
│   └── Dockerfile
├── frontend/
│   ├── app.py                     ← dashboard + ML risk calculator tab
│   ├── requirements.txt
│   └── Dockerfile
├── nginx/
│   ├── nginx.conf                 ← reverse proxy + Basic Auth (/dagster/)
│   └── .htpasswd
└── ml/
    ├── train_model.py             ← trains models, logs to MLflow, saves best_model.pkl
    ├── Dockerfile                 ← image for the ml-trainer one-off service
    ├── mlflow.Dockerfile          ← image for the mlflow tracking server
    └── requirements.txt
```

Two named Docker volumes support the ML workflow:
`model_artifacts` (shared between `ml-trainer` and `fastapi`, holds
`best_model.pkl` and `model_meta.json`) and `mlflow_data` (MLflow's
sqlite backend store and experiment artifacts).
