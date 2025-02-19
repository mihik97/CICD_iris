import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_data('data/iris_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)