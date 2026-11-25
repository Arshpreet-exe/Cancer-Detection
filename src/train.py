import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


def main():
    # Load dataset
    data_path = os.path.join("data", "cancer_data.csv")
    dataset = pd.read_csv(data_path)

    # Features (exclude Sample code number at index 0 and Class target at the end)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Train Logistic Regression model
    classifier = LogisticRegression(random_state=0, max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # 10-Fold Cross Validation
    accuracies = cross_val_score(
        estimator=classifier, X=X_train, y=y_train, cv=10
    )

    print("Confusion Matrix:\n", cm)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"10-Fold CV Accuracy: {accuracies.mean() * 100:.2f}% (+/- {accuracies.std() * 100:.2f}%)")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "logistic_model.pkl")
    joblib.dump(classifier, model_path)
    print(f"Model saved successfully to '{model_path}'")


if __name__ == "__main__":
    main()
