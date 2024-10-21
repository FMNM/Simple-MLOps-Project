from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
project_id = os.getenv("GCC_PROJECT_ID")
location = os.getenv("GCC_LOCATION")
bucket = os.getenv("GCC_BUCKET_NAME")
