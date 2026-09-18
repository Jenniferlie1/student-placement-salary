import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from preprocessing import preprocess
from evaluation import evaluate


def train():
    mlflow.set_tracking_uri("file:./mlruns")
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment("Student-Pipeline")

    # Load data
    train_class, test_class, train_reg, test_reg = preprocess()

    x_train_class, y_train_class = train_class
    x_test_class, y_test_class = test_class

    x_train_reg, y_train_reg = train_reg
    x_test_reg, y_test_reg = test_reg

    # Define Preprocessor
    num_cols = x_train_class.select_dtypes(include='number').columns
    cat_cols = x_train_class.select_dtypes(exclude='number').columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    with mlflow.start_run() as run:

        # Classification Pipeline
        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVC())
        ])
        clf.fit(x_train_class, y_train_class)

        # Regression Pipeline
        reg = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42))
        ])
        reg.fit(x_train_reg, y_train_reg)

        # Log Parameters
        mlflow.log_param("classification_model", "SVC")
        mlflow.log_param("regression_model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("random_state", 42)

        # Log Metrics (Train)
        mlflow.log_metric("train_accuracy", clf.score(x_train_class, y_train_class))
        mlflow.log_metric("train_r2", reg.score(x_train_reg, y_train_reg))

        # Log Metrics (Test)
        mlflow.log_metric("test_accuracy", clf.score(x_test_class, y_test_class))
        mlflow.log_metric("test_r2", reg.score(x_test_reg, y_test_reg))

        # Log Models (Mlflow)
        mlflow.sklearn.log_model(clf, "classification-model")
        mlflow.sklearn.log_model(reg, "regression-model")

        # Save Local Artifacts
        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(clf, "artifacts/classification_model.pkl")
        joblib.dump(reg, "artifacts/regression_model.pkl")

        print("\nRun ID:", run.info.run_id)

        return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    evaluate(run_id)