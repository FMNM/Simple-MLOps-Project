# hyperparameter_tuning.py
import os
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from custom_utils import bucket, project_id, location

# Initialize Vertex AI
aiplatform.init(project=project_id, location=location, staging_bucket=bucket)

# Define the hyperparameter tuning job
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="xgboost-hp-tuning-job",
    custom_job=aiplatform.CustomJob(
        display_name="xgboost-hp-tuning",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": f"gcr.io/{project_id}/xgboost-trainer:latest",
                    "env": [
                        {"name": "GCP_PROJECT_ID", "value": project_id},
                        {"name": "GCP_LOCATION", "value": location},
                        {"name": "GCP_BUCKET_NAME", "value": bucket},
                    ],
                    "args": [
                        # "--bucket_name",
                        # bucket,
                        # "--trial_id",
                        # "{trial_id}",
                    ],
                },
            }
        ],
    ),
    metric_spec={"r2_score": "maximize"},
    parameter_spec={
        "n_estimators": hpt.IntegerParameterSpec(min=50, max=100, scale="UNIT_LINEAR_SCALE"),
        "max_depth": hpt.IntegerParameterSpec(min=3, max=5, scale="UNIT_LINEAR_SCALE"),
        "learning_rate": hpt.DoubleParameterSpec(min=0.01, max=0.1, scale="UNIT_LINEAR_SCALE"),
        "subsample": hpt.DoubleParameterSpec(min=0.8, max=1.0, scale="UNIT_LINEAR_SCALE"),
    },
    max_trial_count=4,
    parallel_trial_count=4,
)

# Run the hyperparameter tuning job
hp_job.run()

# Wait for the job to complete and gather trials
hp_job.wait()
trials = hp_job.trials

# Save all trial results to a JSON file
trials_data = [trial.to_dict() for trial in trials]
with open("trials.json", "w") as f:
    json.dump(trials_data, f, indent=4)

# Get the best trial by R² score
best_trial = max(trials, key=lambda t: t.final_measurement.metrics["r2_score"].value)
print(f"Best Trial ID: {best_trial.id}")
print(f"Best Hyperparameters: {best_trial.parameters}")
