import argparse
import joblib
from sklearn.datasets import load_iris

def predict_flower(model_path, sample):
    iris = load_iris()
    model = joblib.load(model_path)

    prediction = model.predict([sample])[0]
    print("Predicted class:", iris.target_names[prediction])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/iris_model.joblib")
    parser.add_argument("--sepal-length", type=float, required=True)
    parser.add_argument("--sepal-width", type=float, required=True)
    parser.add_argument("--petal-length", type=float, required=True)
    parser.add_argument("--petal-width", type=float, required=True)

    args = parser.parse_args()

    sample = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width
    ]

    predict_flower(args.model_path, sample)
