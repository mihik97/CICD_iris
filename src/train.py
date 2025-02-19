import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess_data            #loading from preprocessing script

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=420)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    data = load_data('data/iris_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    joblib.dump(model, 'model/random_forest_model.pkl')