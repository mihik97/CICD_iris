import joblib
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_data                #loading from preprocessing script

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    data = load_data('data/iris_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = joblib.load('model/random_forest_model.pkl')
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")