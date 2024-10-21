# custom_utils.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION")
bucket = os.getenv("GCP_BUCKET_NAME")
