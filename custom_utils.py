from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
project_id = os.getenv("GCC_PROJECT_ID")
hpt_job_id = os.getenv("GCC_HPO_JOB_ID")
bucket = os.getenv("GCC_BUCKET_NAME")
location = os.getenv("GCC_LOCATION")
