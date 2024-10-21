from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from custom_utils import bucket, hpt_job_id, location, project_id

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location)

# Define hyperparameter search space
parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.1, scale="linear"),
    "max_depth": hpt.IntegerParameterSpec(min=3, max=6, scale="linear"),
    "n_estimators": hpt.IntegerParameterSpec(min=10, max=20, scale="linear"),
    "subsample": hpt.DoubleParameterSpec(min=0.8, max=1.0, scale="linear"),
}

# Define the metric to optimize
metric_spec = {"MSE": "minimize"}

# Create the Custom Job
custom_job = aiplatform.CustomJob(
    staging_bucket=f"gs://{bucket}",
    display_name="xgb_housing_job",
    worker_pool_specs=[
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": f"gcr.io/{project_id}/xgb_trainer:latest",
            },
        }
    ],
)

# Create the Hyperparameter Tuning Job
hpt_job = aiplatform.HyperparameterTuningJob(
    display_name="xgb_housing_hpt",
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=10,
    parallel_trial_count=4,
)

# Run the Hyperparameter Tuning Job
hpt_job.run()
