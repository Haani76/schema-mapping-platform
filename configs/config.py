import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project
    PROJECT_NAME = os.getenv("PROJECT_NAME", "schema-mapping-platform")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Model
    MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "schema-mapping-ner")

    # AWS
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training")

    # NER Labels
    LABELS = [
        "O",
        "B-CUSTOMER_ID",
        "B-PRODUCT_ID",
        "B-REVENUE",
        "B-DATE",
        "B-QUANTITY",
        "B-LOCATION",
        "B-EMAIL",
        "B-PHONE",
        "B-NAME",
        "B-STATUS",
        "B-CATEGORY",
    ]

    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
    NUM_LABELS = len(LABELS)


config = Config()