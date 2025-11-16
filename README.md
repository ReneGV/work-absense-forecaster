work-absense-forecaster
==============================

![Tests](https://github.com/renegv/work-absense-forecaster/actions/workflows/tests.yml/badge.svg)

A work absense hours ML project

## 📑 Table of Contents

- [Setup](#setup)
- [🚀 API Server](#-api-server)
  - [Running with Docker](#running-with-docker-recommended)
  - [API Endpoints](#api-endpoints)
  - [Testing the API](#testing-the-api)
  - [Interactive Documentation](#interactive-documentation)
- [Project Organization](#project-organization)
- [🧪 Testing Guide](#-testing-guide-for-work-absenteeism-forecaster)
  - [Test Structure](#-test-structure)
  - [Test Coverage](#test-coverage)
  - [Running Tests](#-running-tests)
  - [Continuous Integration](#-continuous-integration)
- [MLflow Integration](#mlflow-integration)

---

## Setup

```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## MLflow Integration

Run MLFlow server for experiment tracking:

```bash
mlflow server --backend-store-uri sqlite:///my.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

The MLflow UI will be available at `http://localhost:5000`

## 🚀 API Server

The project includes a FastAPI-based REST API for making absenteeism predictions.

### Running with Docker (Recommended)

**Build and start the API:**
```bash
docker-compose up -d
```

**Stop the API:**
```bash
docker-compose down
```

### API Endpoints

The API will be available at `http://localhost:8000`

- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /predict** - Make absenteeism predictions
- **GET /docs** - Interactive Swagger UI documentation

### Testing the API

**Health check:**
```bash
curl http://localhost:8000/health
```

**Get API information:**
```bash
curl http://localhost:8000/
```

**Make a prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reason_for_absence": 23,
    "month_of_absence": 7,
    "day_of_the_week": 3,
    "seasons": 1,
    "transportation_expense": 289,
    "distance_from_residence_to_work": 36,
    "service_time": 13,
    "age": 33,
    "work_load_average/day": 239.554,
    "hit_target": 97,
    "disciplinary_failure": 0,
    "education": 1,
    "son": 2,
    "social_drinker": 1,
    "social_smoker": 0,
    "pet": 1,
    "weight": 90,
    "height": 172
  }'
```

**Expected response:**
```json
{
  "prediction": 1,
  "prediction_label": "High",
  "confidence": 0.7343
}
```

### Interactive Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all available endpoints
- See request/response schemas
- Test the API directly from your browser
- Download OpenAPI specification


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── Dockerfile         <- Docker configuration for the API server
    ├── docker-compose.yml <- Docker Compose configuration for easy deployment
    ├── requirements.txt   <- Full requirements file for development and training
    ├── requirements-api.txt <- Minimal requirements for the API server
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- FastAPI REST API for predictions
    │   │   └── server.py  <- API server implementation
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    │   │   └── preprocessors.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests              <- Unit and integration tests
    │   ├── unit/          <- Unit tests for individual components
    │   └── integration/   <- End-to-end integration tests
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## 🧪 Testing Guide for Work Absenteeism Forecaster

### 📋 Test Structure

A comprehensive test suite with both **unit tests** and **integration tests** has been created:

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Shared pytest fixtures (root level)
│
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── conftest.py               # Unit-specific fixtures
│   ├── test_preprocessors.py     # Tests for custom transformers
│   ├── test_train_model.py       # Tests for training pipeline
│   ├── test_predict_model.py     # Tests for prediction pipeline
│   ├── test_data_utils.py        # Tests for data utilities
│   └── test_evaluation.py        # Tests for model evaluation
│
└── integration/                   # Integration tests
    ├── __init__.py
    ├── conftest.py                # Integration-specific fixtures
    └── test_pipeline_integration.py  # End-to-end pipeline tests

Additional files:
├── pytest.ini                     # Pytest configuration
├── Dockerfile.test                # Docker image for testing
└── docker-compose.test.yml        # Docker compose for running tests
```

### Test Coverage

#### Unit Tests (`tests/unit/`)

1. **Preprocessors** (`test_preprocessors.py`)
   - `DropColumnsTransformer`: Column dropping functionality
   - `IQRClippingTransformer`: Outlier handling using IQR method
   - `ToStringTransformer`: Type conversion to strings
   - Integration with sklearn pipelines

2. **Model Training** (`test_train_model.py`)
   - Data loading and preparation
   - Pipeline construction
   - Model creation (Logistic Regression, Random Forest, Neural Network)
   - Training and evaluation
   - Multiple model training
   - Model persistence

3. **Model Prediction** (`test_predict_model.py`)
   - Model loading
   - Making predictions on new data
   - Data handling in prediction pipeline

4. **Data Utilities** (`test_data_utils.py`)
   - CSV file loading
   - Column name normalization
   - Data shape validation
   - Data value preservation

5. **Model Evaluation** (`test_evaluation.py`)
   - Metrics calculation (accuracy, F1, recall, precision)
   - Classification reports
   - Confusion matrix creation

#### Integration Tests (`tests/integration/`)

**End-to-End Pipeline** (`test_pipeline_integration.py`)

1. **Complete ML Workflow** (`test_realistic_ml_workflow`)
   - Data loading and preparation
   - Train/test split
   - Preprocessing pipeline creation
   - Model training
   - Model persistence (save/load)
   - Prediction on new data
   - Metrics evaluation
   - Confusion matrix generation
   - File artifact verification

---

### 🔄 Continuous Integration

This project uses GitHub Actions to automatically run tests on every push and pull request.

#### Workflow Overview

The CI pipeline runs:
- **Unit tests** on all test files in `tests/unit/`
- **Integration tests** on all test files in `tests/integration/`
- **Coverage reports** with XML and HTML output
- Tests on Python 3.9

#### Viewing Test Results

1. Navigate to the **Actions** tab in the GitHub repository
2. Click on any workflow run to see detailed test results
3. Coverage reports are uploaded as artifacts (available for 30 days)

#### Workflow Configuration

The workflow is defined in `.github/workflows/tests.yml` and triggers on:
- Pushes to `main` and `develop` branches
- Pull requests to `main` and `develop` branches

---

### 🚀 Running tests

#### Build Docker Image

Build the docker image that contains the required environment to run the tests:

```sh
docker build -f Dockerfile.test -t work-absenteeism-test:latest .
```

#### Run All Tests

Run all tests (unit + integration):

```sh
docker-compose -f docker-compose.test.yml run --rm test
```

#### Run Specific Test Suites

Run only unit tests:

```sh
docker-compose -f docker-compose.test.yml run --rm test pytest tests/unit/ -v
```

Run only integration tests:

```sh
docker-compose -f docker-compose.test.yml run --rm test pytest tests/integration/ -v
```

#### Run with Coverage

Run tests with coverage report:

```sh
docker-compose -f docker-compose.test.yml run --rm test-coverage
```

#### Test Markers

Use pytest markers to run specific test categories:

```sh
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only slow tests
pytest -m slow
```
