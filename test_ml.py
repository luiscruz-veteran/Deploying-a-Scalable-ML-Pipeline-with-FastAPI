import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


@pytest.fixture
def sample_data():
    """Creates a small deterministic sample dataset for testing."""
    data = pd.DataFrame({
        "age": [39, 50, 38, 53],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
        "education-num": [13, 13, 9, 7],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Handlers-cleaners"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
        "race": ["White", "White", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "capital-gain": [2174, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": ["<=50K", "<=50K", "<=50K", ">50K"],
    })
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return data, cat_features


def test_process_data(sample_data):
    """Tests that process_data returns expected array shapes and types."""
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]


def test_train_and_inference(sample_data):
    """Tests that train_model fits a classifier and inference outputs valid predictions."""
    data, cat_features = sample_data
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape


def test_compute_model_metrics():
    """Tests that compute_model_metrics returns exact expected scores."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
