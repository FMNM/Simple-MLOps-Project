from google.cloud import storage, aiplatform
from custom_utils import location, project_id, bucket
import xgboost as xgb
import pandas as pd

aiplatform.init(project=project_id, location=location)
storage_client = storage.Client()

# Download the best model
bucket = storage_client.bucket(bucket)
blob = bucket.blob("xgboost-hpt/best_trial/model.bst")
blob.download_to_filename("model.bst")

# Load the model
model = xgb.Booster()
model.load_model("model.bst")

# Prepare instances for prediction
X_val = pd.read_csv("data/X_val.csv").head(5)
dtest = xgb.DMatrix(X_val)
predictions = model.predict(dtest)

print("Predictions:", predictions)
