# Local Development with Docker

**Prerequisites**:

- Python 3.12
- Docker & Docker Compose
- Git + DVC
- (Optional) Google Cloud CLI for local testing

1. **Clone the Repository**

```bash
git clone https://github.com/roissyahf/mlops-loan-approval
cd mlops-loan-approval
```

2. **Set up virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure DVC remote**

You must configure DVC to know where the data is stored—your DagsHub remote. You will be prompted for your DagsHub Username and DagsHub Token when DVC needs to pull the data.

```bash
dvc remote add dagshub_remote https://dagshub.com/roissyahfk/Loan_Approval_Model.dvc
dvc remote default dagshub_remote
```

5. **Pull the data**

```bash
dvc pull
```

6. **Run `model/modelling_refactor`**

By running the script, you will get `model.pkl` locally. It's crucial for successful model, api, and frontend app running.

7. **Run services**

```bash
docker compose up --build
```

> In local development, all services (frontend, API, model, monitoring) can run together with Docker Compose.

8. **Access Services**

- Model: `http://localhost:5000/health`
- API: `http://localhost:8000/health`
- Frontend: `http://localhost:3001`
- Monitoring: `http://localhost:5003`


---

# Production (CI/CD + GCP)

**Prerequisites**:

- Google Cloud Project with billing enabled
- Artifact Registry + Cloud Run APIs enabled
- Service Account JSON (CI/CD)

**Setup**

1. **Authenticate**

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
```


2. **Enable APIs**

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```


3. **Create Artifact Registry**

```bash
gcloud artifacts repositories create loan-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker repo for loan system"
```


4. **Service Accounts**

```bash
gcloud iam service-accounts create sa-model \
  --display-name="Model Service Account"
```

Give roles (Cloud Run Admin, Artifact Registry Writer, etc.).

5. **Configure GitHub Secrets and Environment variables**

Add all required keys under repo → Settings → Secrets and variables → Actions
`GCP_PROJECT_ID`, `GCP_REGION`, `ARTIFACT_REPO`, `GCP_SA_KEY`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`

Environment variables injected at deploy:
   `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `GCP_PROJECT_ID`

6. **CI/CD Workflows**

Each service has its own workflow under `.github/workflows/`:
- api.yml → builds & deploys API service
- model.yml → builds & deploys Model service (fetches model.pkl from MLflow/DVC)
- frontend.yml → builds & deploys Frontend
- monitoring.yml → builds & deploys Monitoring
  
7. **Deployment Targets**

* `loan-frontend` → UI
* `loan-api` → Flask API
* `loan-model` → ML inference (model pulled from MLflow Registry)
* `loan-monitor` → Evidently drift monitoring

---

# Dev vs Prod Matrix

| Service        | Local entrypoint                    | Prod entrypoint (Cloud Run) | Dockerfile(s)                                          | Artifacts       | Key env vars             |
| -------------- | ----------------------------------- | --------------------------- | ------------------------------------------------------ | --------------- | ------------------------ |
| **Frontend**   | `frontend/app_dev.py` or static         | `loan-frontend`             | `Dockerfile.local.frontend`, `Dockerfile.frontend`     | —               | `API_URL`                 |
| **API**        | `api/app_dev.py`                        | `loan-api`                  | `Dockerfile.local.api`, `Dockerfile.api`               | Uses Model service  | `MODEL_URL`      |
| **Model**      | `model/app_dev.py`                      | `loan-model`                | `Dockerfile.local.model`, `Dockerfile.model`           | MLflow Registry | `MLFLOW_*` |
| **Monitoring** | `monitoring/app_dev.py`                 | `loan-monitor`              | `Dockerfile.local.monitoring`, `Dockerfile.monitoring` | DVC data + logs | —                    |

