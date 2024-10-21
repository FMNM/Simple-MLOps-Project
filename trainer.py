import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import joblib
import os


def main(args):
    # Load data
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="MedHouseVal")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate, subsample=args.subsample, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    preds = model.predict(X_val)
    mse = root_mean_squared_error(y_val, preds)

    # Save model and metrics
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "model.joblib"))
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse}\n")

    # Report metric to Vertex AI
    print(f"MSE={mse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    main(args)
