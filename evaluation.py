import mlflow
from sklearn.metrics import accuracy_score, r2_score

from preprocessing import preprocess


def evaluate(run_id):

    # Load from preprocessing
    train_class, test_class, train_reg, test_reg = preprocess()

    x_test_class, y_test_class = test_class
    x_test_reg, y_test_reg = test_reg

    # Load models from MLflow
    clf = mlflow.sklearn.load_model(
        f"runs:/{run_id}/classification-model"
    )

    reg = mlflow.sklearn.load_model(
        f"runs:/{run_id}/regression-model"
    )

    # Predict
    y_pred_class = clf.predict(x_test_class)
    y_pred_reg = reg.predict(x_test_reg)

    # Evaluate
    accuracy = accuracy_score(y_test_class, y_pred_class)
    r2 = r2_score(y_test_reg, y_pred_reg)

    print("\n=== Evaluation Result ===")
    print("Accuracy (Classification):", accuracy)
    print("R2 Score (Regression):", r2)

    return accuracy, r2


if __name__ == "__main__":
    evaluate("FILL_RUN_ID")