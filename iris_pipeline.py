import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optionnel pour un meilleur style des graphiques
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


def load_data():
    """Load the Iris dataset into a pandas DataFrame."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y, iris.target_names


def visualize_data(X, y):
    """Visualize the distribution and relationships of the features."""
    # Histogram for each feature
    X.hist(bins=15, figsize=(10, 8))
    plt.suptitle("Feature Distributions")
    plt.show()

    # Scatter plot of two important features
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Sepal length vs Sepal width")
    plt.show()


def train_model(X, y):
    """Train a Logistic Regression model and evaluate it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200, solver="liblinear")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Model Accuracy:", round(accuracy, 3))
    print("\nClassification Report:\n", report)

    # Visualize confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Save the model
    joblib.dump(model, "iris_logistic_regression_model.pkl")
    print("Model saved as 'iris_logistic_regression_model.pkl'")

    return model


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix to evaluate the model."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Setosa", "Versicolor", "Virginica"], yticklabels=["Setosa", "Versicolor", "Virginica"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    X, y, target_names = load_data()
    visualize_data(X, y)
    train_model(X, y)