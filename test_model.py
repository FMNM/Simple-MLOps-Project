from google.cloud import aiplatform
from data_preparation import load_data
import pandas as pd
from custom_utils import location, project_id

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location)

# Load data
X, _ = load_data()
instances = X.iloc[:5].to_dict(orient="records")

# Get the endpoint
endpoints = aiplatform.Endpoint.list(filter="display_name='xgb_housing_model'")
endpoint = endpoints[0]

# Make predictions
predictions = endpoint.predict(instances)
print("Predictions:", predictions)
