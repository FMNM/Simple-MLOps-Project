from google.cloud import aiplatform
from custom_utils import bucket, hpt_job_id, location, project_id

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location)

# Get the hyperparameter tuning job
hpt_job_name = f"projects/{project_id}/locations/{location}/hyperparameterTuningJobs/{hpt_job_id}"
hpt_job = aiplatform.HyperparameterTuningJob(hpt_job_name)

# Get the best trial
best_trial = hpt_job.trials[0]
print("Best Trial Parameters:", best_trial.parameters)
print("Best Trial MSE:", best_trial.final_measurement.metrics["MSE"])

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
