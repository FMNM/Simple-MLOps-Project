import os
import argparse
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from google.cloud import storage
from sklearn.datasets import fetch_california_housing
from google.cloud import aiplatform


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Received arguments: {args}")

    # Retrieve environment variables
    try:
        bucket = os.environ["GCP_BUCKET_NAME"]
        project_id = os.environ["GCP_PROJECT_ID"]
        location = os.environ["GCP_LOCATION"]
    except KeyError as e:
        logger.error(f"Environment variable {e} not set.")
        raise

    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # Load data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model with hyperparameters
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=42,
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    # Report metrics
    logger.info(f"r2_score: {r2}")

    # Report metrics in JSON format for Vertex AI
    # Vertex AI captures metrics when printed in the following JSON format
    print(f'{{"metric": "r2_score", "value": {r2}}}')

    # Save metrics to a JSON file (optional)
    metrics = {
        "r2_score": r2,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
        },
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Save and upload the model
    model.save_model("model.bst")
    upload_model(bucket, "model.bst", "models/best_model.bst")

    # Deploy the model using Vertex AI
    deploy_model_to_vertex_ai(bucket, project_id, location)


def upload_model(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"Model {source_file_name} uploaded to {destination_blob_name}.")


def deploy_model_to_vertex_ai(bucket, project_id, location):
    """Deploys the trained model to Vertex AI."""
    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    model = aiplatform.Model.upload(
        display_name="xgboost-regression-model",
        artifact_uri=f"gs://{bucket}/models/best_model.bst",
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-4:latest",
    )

    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
    )
    print(f"Model deployed to endpoint: {endpoint.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--subsample", type=float, required=True)
    args = parser.parse_args()
    main(args)
