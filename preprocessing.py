import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess():
    # Load data
    df = pd.read_csv("ingested/B.csv")

    # Split features & target
    x = df.drop(['placement_status', 'salary_package_lpa', 'student_id'], axis=1)
    y_class = df['placement_status']
    y_reg = df['salary_package_lpa']

    # Train test split
    x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        x, y_class, y_reg,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    train_class = (x_train, y_train_class)
    test_class = (x_test, y_test_class)

    train_reg = (x_train, y_train_reg)
    test_reg = (x_test, y_test_reg)

    return train_class, test_class, train_reg, test_reg


if __name__ == "__main__":
    preprocess()