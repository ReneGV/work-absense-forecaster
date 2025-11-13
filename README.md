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

## 📋 What Has Been Created

A comprehensive test suite for the `src/models` module has been created with the following structure:

```
tests/
├── __init__.py                    # Package initialization
├── conftest.py                    # Shared pytest fixtures
├── test_preprocessors.py          # Tests for custom transformers (30+ tests)
├── test_train_model.py           # Tests for training pipeline (15+ tests)
├── test_predict_model.py         # Tests for prediction pipeline (15+ tests)
├── test_parameter_tuning.py      # Tests for hyperparameter tuning (10+ tests)
├── README.md                      # Test documentation
└── TEST_SUMMARY.md               # Detailed test summary

Additional files:
├── pytest.ini                     # Pytest configuration
├── run_tests.sh                   # Convenience script for running tests
└── .github/workflows/tests.yml    # CI/CD workflow for automated testing
```

## Test Coverage

The test suite covers:

1. **Preprocessors (`test_preprocessors.py`)**
   - `DropColumnsTransformer`: Column dropping functionality
   - `IQRClippingTransformer`: Outlier handling using IQR method
   - `ToStringTransformer`: Type conversion to strings
   - Integration with sklearn pipelines

2. **Model Training (`test_train_model.py`)**
   - Pipeline construction
   - Model creation (Logistic Regression, Random Forest, Neural Network)
   - Data preparation and preprocessing
   - Training process and metrics calculation

3. **Model Prediction (`test_predict_model.py`)**
   - Model loading and saving
   - Making predictions on new data
   - Prediction evaluation with ground truth
   - Data handling in prediction pipeline

4. **Parameter Tuning (`test_parameter_tuning.py`)**
   - Parameter grid setup
   - GridSearchCV functionality
   - Best model selection
   - Metrics tracking during tuning

---

## 🚀 Quick Start

Build the docker image that conatins the required environment to run the tests.

```sh
docker build -f Dockerfile.test -t work-absenteeism-test:latest .
```

Run tests inside the docker container
```sh
docker-compose -f docker-compose.test.yml run --rm test
```
