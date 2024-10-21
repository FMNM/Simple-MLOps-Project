from google.cloud import aiplatform
from custom_utils import bucket, hpt_job_id, location, project_id

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location)

# Get the hyperparameter tuning jobs
hpt_jobs = aiplatform.HyperparameterTuningJob.list(order_by="create_time desc")

# Check if there are any hyperparameter tuning jobs
if not hpt_jobs:
    raise ValueError("No hyperparameter tuning jobs found.")

# Get the latest hyperparameter tuning job
latest_hpt_job = hpt_jobs[0]

# Get the best trial
best_trial = latest_hpt_job.trials[0]
print("Best Trial Parameters:", best_trial.parameters)

# Print all available metrics in the best trial's final measurement
print("Available Metrics in Best Trial's Final Measurement:")
for metric in best_trial.final_measurement.metrics:
    print(f"Metric ID: {metric.metric_id}, Value: {metric.value}")

# Find the MSE metric in the best trial's final measurement
mse_metric = None
for metric in best_trial.final_measurement.metrics:
    if metric.metric_id == "MSE":
        mse_metric = metric.value
        break

if mse_metric is None:
    raise ValueError("MSE metric not found in the best trial's final measurement.")

print("Best Trial MSE:", mse_metric)

# Assuming the model is saved to Cloud Storage during training
model_artifact_uri = f"gs://{bucket}/{hpt_job_id}"

# Upload the model to Vertex AI
model = aiplatform.Model.upload(
    display_name="xgb_housing_model",
    artifact_uri=model_artifact_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-3:latest",
)

# Deploy the model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
)
