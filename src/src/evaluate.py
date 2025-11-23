import argparse
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(model_path):
    iris = load_iris()
    X, y = iris.data, iris.target

    model = joblib.load(model_path)
    predictions = model.predict(X)

    print(classification_report(y, predictions, target_names=iris.target_names))

    cm = confusion_matrix(y, predictions)

    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("reports/confusion_matrix.png")

    print("Confusion matrix saved to reports/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/iris_model.joblib")
    args = parser.parse_args()

    evaluate_model(args.model_path)
