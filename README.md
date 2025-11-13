work-absense-forecaster
==============================

A work absense hours ML project

Setup

```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

Run MLFlow server
```
mlflow server  --backend-store-uri sqlite:///my.db   --default-artifact-root ./mlruns   --host 0.0.0.0 --port 5000
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Testing Guide for Work Absenteeism Forecaster

## 📋 Test Structure

A comprehensive test suite with both **unit tests** and **integration tests** has been created:

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Shared pytest fixtures (root level)
│
├── unit/                          # Unit tests (55 tests)
│   ├── __init__.py
│   ├── test_preprocessors.py      # Tests for custom transformers
│   ├── test_train_model.py       # Tests for training pipeline
│   ├── test_predict_model.py     # Tests for prediction pipeline
│   └── test_parameter_tuning.py  # Tests for hyperparameter tuning
│
└── integration/                   # Integration tests (21 tests)
    ├── __init__.py
    ├── conftest.py                # Integration-specific fixtures
    └── test_pipeline_integration.py  # End-to-end pipeline tests

Additional files:
├── pytest.ini                     # Pytest configuration
├── Dockerfile.test                # Docker image for testing
└── docker-compose.test.yml        # Docker compose for running tests
```

## Test Coverage

### Unit Tests (`tests/unit/`)

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

### Integration Tests (`tests/integration/`)

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

## 🚀 Quick Start

### Build Docker Image

Build the docker image that contains the required environment to run the tests:

```sh
docker build -f Dockerfile.test -t work-absenteeism-test:latest .
```

### Run All Tests

Run all tests (unit + integration):

```sh
docker-compose -f docker-compose.test.yml run --rm test
```

### Run Specific Test Suites

Run only unit tests:

```sh
docker-compose -f docker-compose.test.yml run --rm test pytest tests/unit/ -v
```

Run only integration tests:

```sh
docker-compose -f docker-compose.test.yml run --rm test pytest tests/integration/ -v
```

### Run with Coverage

Run tests with coverage report:

```sh
docker-compose -f docker-compose.test.yml run --rm test-coverage
```

### Test Markers

Use pytest markers to run specific test categories:

```sh
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only slow tests
pytest -m slow
```

## 📊 Test Statistics

- **Total Tests**: 76
- **Unit Tests**: 55
- **Integration Tests**: 21
- **Test Coverage**: ~85% of `src/models` module
