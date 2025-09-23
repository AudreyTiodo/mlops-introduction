import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Pour sauvegarder le modèle

def load_data():
    """Load the Iris dataset into a pandas DataFrame."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y

def train_model(X, y):
    """Train a Logistic Regression model and evaluate it."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=load_iris().target_names)

    print("Model Accuracy:", round(accuracy, 3))
    print("\nClassification Report:\n", report)

    # Save the model
    joblib.dump(model, "iris_logistic_regression_model.pkl")
    print("Model saved as 'iris_logistic_regression_model.pkl'")

    return model

if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)