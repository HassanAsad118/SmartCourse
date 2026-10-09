import os

# -----------------------------------------------
# BASE DIRECTORY
# Absolute path to the project root directory.
# All other paths are built relative to this,
# ensuring portability across Windows and Linux.
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------
# DATA PATHS
# -----------------------------------------------
DATA_FILE = os.path.join(BASE_DIR, "data", "Courses_dataset.csv")


# -----------------------------------------------
# MODEL PATHS
# -----------------------------------------------
TFIDF_MODEL_PATH  = os.path.join(BASE_DIR, "models", "tfidf_model.pkl")
NEURAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "neural_embeddings.pkl")


# -----------------------------------------------
# DATABASE PATH
# -----------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "database.db")


# -----------------------------------------------
# FLASK SETTINGS
# Replace SECRET_KEY with an environment variable
# before deploying to any public server.
# -----------------------------------------------
SECRET_KEY = os.environ.get("SMARTCOURSE_SECRET_KEY", "smartcourse-secret-key")


# -----------------------------------------------
# RECOMMENDATION CONSTANTS
# -----------------------------------------------
TOP_K        = 10                            # Number of results returned per query
MODEL_TYPES  = ["tfidf", "neural", "both"]  # Valid model type values for API