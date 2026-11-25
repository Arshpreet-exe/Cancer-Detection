import os
import joblib
import pandas as pd


def main():
    model_path = os.path.join("models", "logistic_model.pkl")
    input_path = os.path.join("data", "new_datasets.csv")
    output_path = os.path.join("data", "new_dataset_with_predictions.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run 'python src/train.py' first."
        )

    classifier = joblib.load(model_path)
    new_dataset = pd.read_csv(input_path)

    # Ensure new_datasets.csv matches the 9 feature columns used during training
    X_new = new_dataset.values
    y_pred = classifier.predict(X_new)

    df_with_predictions = new_dataset.copy()
    df_with_predictions["Predicted_Target"] = y_pred
    df_with_predictions.to_csv(output_path, index=False)

    print("Predicted Output:\n", y_pred)
    print(f"Predictions saved to '{output_path}'")


if __name__ == "__main__":
    main()
