import xgboost as xgb
import argparse
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=50)
parser.add_argument("--max_depth", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--subsample", type=float, default=0.8)
args = parser.parse_args()

# Load the California Housing dataset directly from scikit-learn
data = fetch_california_housing()
X, y = data.data, data.target

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    subsample=args.subsample,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
