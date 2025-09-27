# MLOps Loan Approval Prediction System

A production-ready ML system that predicts **loan approval** using applicant information such as income, age, and credit score.

The project demonstrates an **end-to-end MLOps workflow**: data versioning, model training, experiment tracking, monitoring, CI/CD, and deployment on Google Cloud Run.

---

## 📊 Dataset

The system is trained on the [Loan Approval Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data).

Features include:

* Applicant age, income, credit score
* Loan amount, loan interest, loan percent income
* Previous loan default indicator

**Target variable:** loan approval (1 = approved, 0 = rejected)

---

## 🛠️ Technology Stack

* **ML / Training**: scikit-learn, XGBoost
* **Versioning**: DVC (data), MLflow (models) with DagsHub
* **Monitoring**: Evidently, Cloud Logging, Cloud Monitoring, Bigquery
* **Infrastructure**: Docker, Google Cloud Run, Artifact Registry
* **CI/CD**: GitHub Actions
* **Frontend**: HTML/CSS/JS

---

## 🚀 Key Features

* ✅ Microservices architecture (Frontend, API, Model, Monitoring)
* ✅ Data versioning with **DVC** + DagsHub remote storage
* ✅ Model tracking with **MLflow** + DagsHub Registry
* ✅ Drift detection using **Evidently**
* ✅ Containerized services with **Docker & Docker Compose**
* ✅ CI/CD with GitHub Actions → **Cloud Run deployment**
* ✅ Structured logs for prediction audit trail
* ✅ Automated monthly retraining via scheduled GitHub Action

---

## 📈 Development Phases

1. **Phase 1** – Modularization & Local Setup
   Split services, added Dockerfiles, and tested with Docker Compose.

2. **Phase 2** – Versioning & Experiment Tracking
   Data with **DVC + DagsHub**, model runs with **MLflow**.

3. **Phase 3** – Monitoring & Drift Detection
   Integrated Evidently reports, structured logging, Cloud Monitoring.

4. **Phase 4** – CI/CD Automation
   GitHub Actions workflows to build → push → deploy services on Cloud Run.

5. **Phase 5** – Continuous Learning
   Automated retraining pipeline (scheduled monthly) → pushes new models to MLflow → triggers redeployment of model service.

---

## 🏗️ Architecture Overview

### Data Flow

* **Prediction**: User → Frontend → API → Model
* **Logging**: API logs → Cloud Logging → BigQuery
* **Retraining**: Logs + training data → retraining pipeline → MLflow Registry → redeployed model service
* **Monitoring**: Cloud Monitoring for system metrics, Evidently for drift reports

### Automated Workflows

* **Retraining**: Monthly scheduled job (GitHub Actions)
* **Deployment**: On push to main branch
* **Monitoring**: Manual review of Evidently reports + Cloud Monitoring alerts

---

## 📂 Project Structure

```
.
├───.github/workflows							
│           api.yml						# API service CI/CD workflow
│           dvc-push.yml				# DVC CI/CD workflow
│           frontend.yml				# Frontend service CI/CD workflow
│           model.yml					# Model service CI/CD workflow
│           monitoring.yml				# Monitoring service CI/CD workflow
│           orchestrate.yml				# CI/CD workflows orchestration
│           retraining-model.yml		# Model retraining CI/CD workflow
│
├───api										
│       app.py							# Production API service
│       app_dev.py						# Local Development API service
│       requirements.txt				# API service requirements
│
├───data										
│   ├───processed
│   │       retrain_data.csv.dvc		# Retraining data tracked with DVC
│   │       test_data.csv.dvc			# Test data tracked with DVC
│   │       train_data.csv.dvc			# Train data tracked with DVC
│   ├───raw
│   │       loan_data.csv.dvc			# Raw data tracked with DVC
│   └───simulation
│           current.jsonl.dvc			# Prediction result JSONL tracked with DVC
│           current_data.csv.dvc		# Evidently current data tracked with DVC
│           reference_data.csv.dvc		# Evidently reference data tracked with DVC
│
├───docker										
│       Dockerfile.api					# Production Dockerfile API service 
│       Dockerfile.frontend				# Production Dockerfile Frontend service
│       Dockerfile.local.api			# Local Development Dockerfile API service 
│       Dockerfile.local.frontend		# Local Development Dockerfile Frontend service
│       Dockerfile.local.model			# Local Development Dockerfile Model service
│       Dockerfile.local.monitoring		# Local Development Dockerfile Monitoring service
│       Dockerfile.model				# Production Dockerfile Model service
│       Dockerfile.monitoring			# Production Dockerfile Monitoring service
│
├───frontend									
│   │   app.py							# Production Frontend service
│   │   app_dev.py						# Local Development Frontend service
│   │   requirements.txt				# Frontend service requirements
│   ├───static
│   │       style.css					# Frontend styling
│   └───templates
│           index.html					# Production Frontend service code
│           index_dev.html				# Local Development Frontend service code
│
├───model										
│   │   app.py							# Production Model service
│   │   app_dev.py						# Local Development Model service
│   │   convert_logs.py					# Local Development convert logs script
│   │   inference.py					# Production inference script
│   │   inference_dev.py				# Local Development inference script
│   │   model.pkl.dvc					# Local Development model.pkl tracked with DVC
│   │   modelling_refactor.py			# Local Development modelling script
│   │   modelling_tuning.py				# Experiment model tuning script 
│   │   preprocessing_refactor.py		# Experiment data raw preprocessing script
│   │   requirements.txt				# Model service requirements
│   │   retraining_pipeline_dev.py		# Local Development model retraining pipeline script
│   │   retraining_pipeline_prod.py		# Production model retraining pipeline script
│   │   simple_preprocessing.py			# Local Development data preprocessing for model training script
│
└───monitoring									
    │   app.py							# Production Evidently Monitoring service							
│   │   app_dev.py						# Local Development Evidently Monitoring service
│   │   evidently_profile.py			# Evidently Monitoring service profile
│   │   requirements.txt				# Evidently Monitoring service requirements
│
├───docker-compose.yml					# Local Development running services with docker compose
├───requirements.txt					# Requirements for cloning purpose
```

---

## ⚡ Usage

### Local Development with Docker

Ensure you have **Docker** and **Google Cloud CLI** installed.

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

4. **Run services**

```bash
docker compose up --build
```

> In local development, all services (frontend, API, model, monitoring) can run together with Docker Compose.

---

### Production (CI/CD + GCP)

1. **CI/CD Workflows**
   Each service has its own workflow under `.github/workflows/`.

   * Builds → pushes Docker images to Artifact Registry
   * Deploys services to Cloud Run
   * Retraining workflow runs monthly and triggers redeployment

2. **Secrets & Configuration**
   Required secrets:
   `GCP_PROJECT_ID`, `GCP_REGION`, `ARTIFACT_REPO`, `GCP_SA_KEY`,
   `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`

   Environment variables injected at deploy:
   `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `GCP_PROJECT_ID`, `BIGQUERY_DATASET`

3. **Deployment Targets**

   * `loan-frontend` → UI
   * `loan-api` → Flask API
   * `loan-model` → ML inference (model pulled from MLflow Registry)
   * `loan-monitor` → Evidently drift monitoring

---

### Dev vs Prod Matrix

| Service        | Local entrypoint                    | Prod entrypoint (Cloud Run) | Dockerfile(s)                                          | Artifacts       | Key env vars             |
| -------------- | ----------------------------------- | --------------------------- | ------------------------------------------------------ | --------------- | ------------------------ |
| **Frontend**   | `frontend/app_dev.py` or static         | `loan-frontend`             | `Dockerfile.local.frontend`, `Dockerfile.frontend`     | —               | `API_URL`                 |
| **API**        | `api/app_dev.py`                        | `loan-api`                  | `Dockerfile.local.api`, `Dockerfile.api`               | Uses Model service  | `MODEL_URL`      |
| **Model**      | `model/app_dev.py`                      | `loan-model`                | `Dockerfile.local.model`, `Dockerfile.model`           | MLflow Registry | `MLFLOW_*` |
| **Monitoring** | `monitoring/app_dev.py`                 | `loan-monitor`              | `Dockerfile.local.monitoring`, `Dockerfile.monitoring` | DVC data + logs | —                    |
| **Retraining** | `model/retraining_pipeline_dev.py` | GitHub Actions job          | —                                                      | DVC + MLflow    | `MLFLOW_*`, `DAGSHUB_*`  |

---

## 🔁 Reproducibility

* **Data versioning**:

  * Training and retraining datasets tracked with **DVC**, stored remotely in DagsHub.
  * `.dvc` pointer files in Git ensure exact dataset versions can be restored with `dvc pull`.

* **Model versioning**:

  * Models logged and stored in **MLflow Registry** (DagsHub).
  * Each training run saves metrics, parameters, and artifacts for reproducibility.

* **Experiment tracking**:

  * MLflow logs allow to replay experiments and compare runs.
  * Retraining pipeline automatically logs new runs and saves the best model to MLflow.

* **Environment consistency**:

  * Dependencies pinned in `requirements.txt`.
  * Containerized with Docker for consistent runtime between local and production.

Reproduce training with:

```bash
dvc pull data/processed/train_data.csv
python model/retraining_pipeline_prod.py
```

---

## 🔮 Future Work

* [ ] Add Evidently → Cloud Monitoring alerts
* [ ] Trigger retraining on drift detection, not only schedule
* [ ] Automated testing and validation pipelines

---

## 📚 References

* [DVC Docs](https://dvc.org/)
* [MLflow Docs](https://mlflow.org/)
* [Evidently Docs](https://evidentlyai.com/)
* [Drift Concept](https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-prediction-drift)
* [Evidently Community Examples](https://github.com/evidentlyai/community-examples?tab=readme-ov-file)
* [Evidently Usage example](https://medium.com/@pranavk2208/detecting-data-drift-using-evidently-5c8643fd382d)

---

## 👩‍💻 Author

Built by **Roissyah Fernanda**. She acknowledges the use of AI assistance in building this project. If you find issues or want to enhance this project, feel free to open a pull request. Let’s collaborate!