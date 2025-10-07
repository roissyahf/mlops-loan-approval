# MLOps Loan Approval Prediction System

Manual loan approval processes in financial institutions are often slow and inconsistent, making automation both a technical and operational challenge. This project demonstrates how machine learning can automate and standardize such decisions, backed by an **end-to-end MLOps workflow**: data versioning, model training, experiment tracking, monitoring, CI/CD, and deployment on Google Cloud Run.

**Project Constraints:**

- Built as a portfolio project using a GCP free trial, so deployed services will be revoked after the trial period.
- No automated testing yet for model, data, or API components.
- Designed for cost efficiency rather than full-scale production scalability.

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

5. **Phase 5** – Continuous Retraining
   Automated retraining pipeline (scheduled monthly) → pushes new models to MLflow → triggers redeployment of model service.

   > Retraining was originally planned using model predictions stored in `data/simulation/current.jsonl` (for development), and model prediction logs stored in BigQuery (for production). However, this approach was cancelled because those predictions do not contain true ground-truth labels, which makes retraining unreliable and potentially misleading. The project now focuses on Phase 1-4 only.


---

## 🏗️ Architecture Overview

<img width="726" height="636" alt="Image" src="https://github.com/user-attachments/assets/ef815cd6-08a6-4e1a-ab56-3e75bad1f22d" />

### Data Flow

* **Prediction**: User → Frontend → API → Model
* **Logging**: API logs → Cloud Logging → BigQuery
* **Monitoring**: Cloud Monitoring for system metrics, Evidently for drift reports

### Automated Workflows

* **Deployment**: On push to main branch
* **Monitoring**: Manual review of Evidently reports with report periodically updated daily + Cloud Monitoring alerts (only for system metrics)

---

## 📂 Project Structure

```
.
├───.github/workflows							
│           api.yml							# API service CI/CD workflow
│           dvc-push.yml					# DVC CI/CD workflow
│           frontend.yml					# Frontend service CI/CD workflow
│           model.yml						# Model service CI/CD workflow
│           monitoring.yml					# Monitoring service CI/CD workflow
│           orchestrate.yml					# CI/CD workflows orchestration
|           monitoring-periodic-update.yml   # Periodic update of Monitoring service CI/CD workflow
│
├───api										
│       app.py							# Production API service
│       app_dev.py						# Local Development API service
│       requirements.txt				# API service requirements
│
├───data										
│   ├───processed
│   │       test_data.csv.dvc			# Test data tracked with DVC
│   │       train_data.csv.dvc			# Train data tracked with DVC
│   ├───raw
│   │       loan_data.csv.dvc			# Raw data tracked with DVC
│   └───simulation
│           current.jsonl.dvc			# Local Development Prediction result JSONL tracked with DVC
|           prediction_dev_log.csv.dvc  # Local Development Evidently current data tracked with DVC
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
│   │   inference.py					# Production inference script
│   │   inference_dev.py				# Local Development inference script
│   │   model.pkl.dvc					# Local Development model.pkl tracked with DVC
│   │   modelling_refactor.py			# Local Development modelling script
│   │   modelling_tuning.py				# Experiment model tuning script 
│   │   preprocessing_refactor.py		# Experiment data raw preprocessing script
│   │   requirements.txt				# Model service requirements
│
└───monitoring									
│   │   app.py							# Production Evidently Monitoring service							
│   │   app_dev.py						# Local Development Evidently Monitoring service
│   │   convert_dev_logs.py        		# Local Development converting current.jsonl logs
│   │   evidently_profile.py			# Evidently Monitoring service profile
│   │   requirements.txt				# Evidently Monitoring service requirements
│
├───docker-compose.yml					# Local Development running services with docker compose
├───requirements.txt					# Requirements for cloning purpose
```

---

## ⚡ Usage

> 👉 For full local development and production setup instructions, see [COMMAND.md](COMMAND.md).

---

## 🔁 Reproducibility

* **Data versioning**:

  * Training datasets tracked with **DVC**, stored remotely in DagsHub.
  * `.dvc` pointer files in Git ensure exact dataset versions can be restored with `dvc pull`.

* **Model versioning**:

  * Models logged and stored in **MLflow Registry** (DagsHub).
  * Each training run saves metrics, parameters, and artifacts for reproducibility.

* **Experiment tracking**:

  * MLflow logs allow to replay experiments and compare runs.
  * Training pipeline automatically logs new runs and saves the best model to MLflow.

* **Environment consistency**:

  * Dependencies pinned in `requirements.txt`.
  * Containerized with Docker for consistent runtime between local and production.

**To reproduce training:**

```bash
dvc pull data/processed/train_data.csv
python model/modelling_refactor.py
```

Take a look at below section in `model/modelling_refactor.py`

```bash
# Promote to Production stage
client = MlflowClient()
client.transition_model_version_stage(
    name="XGB-best-model-manual", # this is MLFLOW_MODEL_NAME
    version=model_info.registered_model_version,
    stage="Production" # this is MLFLOW_MODEL_STAGE
    )
```

Ensure the following variables are match for production, especially in: `model/inference.py`, `docker/Dockerfile.model`, and also in `.github/workflows/model.yml`
```bash
MLFLOW_MODEL_NAME # give the same value as name
MLFLOW_MODEL_STAGE # give the same value as stage
```

---

## 🔮 Future Work

* [ ] Data, ML Model, and Code Testing
* [ ] Add Evidently → Cloud Monitoring alerts for drift
* [ ] Retraining configuration with ground-truth labels

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

This project was built by Roissyah Fernanda, as part of a personal initiative to explore end-to-end MLOps system design, from development to production deployment.

Some parts of the implementation and documentation were assisted by AI tools (e.g., for code refactoring, formatting, and technical writing support), while all system design, integration, and decision-making were independently developed by the author.

> Contributions, feedback, or collaboration requests are always welcome. Feel free to open an issue or submit a pull request!