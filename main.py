from google.cloud import aiplatform
from custom_utils import location, project_id, bucket

aiplatform.init(project=project_id, location=location, staging_bucket=bucket)


# Define the custom job
custom_job = aiplatform.CustomJob(
    display_name="xgboost-housing-training",
    staging_bucket=f"gs://{bucket}",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 100,
            },
            "container_spec": {
                "image_uri": f"gcr.io/{project_id}/xgboost-trainer:latest",
            },
        }
    ],
)

# Define parameter spec
parameter_spec = {
    "n_estimators": aiplatform.hyperparameter_tuning.IntegerParameterSpec(min=50, max=100, scale="linear"),
    "max_depth": aiplatform.hyperparameter_tuning.IntegerParameterSpec(min=3, max=9, scale="linear"),
    "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0.01, max=0.1, scale="linear"),
    "subsample": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0.8, max=1.0, scale="linear"),
}

# Define hyperparameter tuning job
hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="xgboost-housing-hpt",
    custom_job=custom_job,
    metric_spec={"MSE": "MINIMIZE"},
    parameter_spec=parameter_spec,
    max_trial_count=10,
    parallel_trial_count=3,
)

# Run the job
hpt_job.run()

# Get the path to the best model artifacts
best_model_artifact_uri = f"gs://{bucket}/xgboost-hpt/best_trial/"

# Upload the best model to the Vertex AI Model Registry
model = aiplatform.Model.upload(
    display_name="xgboost-housing-model",
    artifact_uri=best_model_artifact_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
)

print("Model successfully uploaded to the Vertex AI Model Registry.")

# Deploy the model to an endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    endpoint_display_name="xgboost-housing-endpoint",
)

print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")
