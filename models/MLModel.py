import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class MlModel:
    """
    Generic ML model wrapper for training, prediction, and evaluation
    """

    @staticmethod
    def get_model(model_type: str):
        if model_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=42)
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = self.get_model(model_type)

    def split_data(self, data: pd.DataFrame, target: str = 'loan_paid_back', test_size: float = 0.2):
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset!")

        X = data.drop(columns=[target])
        y = data[target].astype(int)

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return acc, cm, report


# ------------------- MAIN EXECUTION -------------------

if __name__ == "__main__":

    DATA_PATH = r"E:\projects\data\train_data_clean.csv"
    TARGET_COL = "loan_paid_back"

    # Load data
    data = pd.read_csv(DATA_PATH)

    model_list = ['decision_tree', 'random_forest']
    results = {}

    for model_name in model_list:
        print("\n" + "=" * 50)
        print(f"Training model: {model_name}")

        ml = MlModel(model_name)

        # Split data
        X_train, X_test, y_train, y_test = ml.split_data(
            data,
            target=TARGET_COL,
            test_size=0.2
        )

        # Train
        ml.train(X_train, y_train)

        # Predict
        y_pred = ml.predict(X_test)

        # Evaluate
        acc, cm, report = ml.evaluate(y_test, y_pred)

        results[model_name] = acc

        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(report)

    print("\n" + "=" * 50)
    print("Final Accuracy Comparison:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
