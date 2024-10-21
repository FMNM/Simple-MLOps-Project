from google.cloud import aiplatform
from custom_utils import bucket

model = aiplatform.Model.upload(
    display_name="xgboost-housing-model",
    artifact_uri=f"gs://{bucket}/xgboost-hpt/best_trial/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
)

# Deploy to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    endpoint_display_name="xgboost-housing-endpoint",
)
