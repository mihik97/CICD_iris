import pytest
from src.preprocess import load_data, preprocess_data              #loading from preprocessing script

def test_load_data():
    data = load_data('data/iris_data.csv')
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/iris_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert len(X_train) > 0
    assert len(X_test) > 0