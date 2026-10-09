import re
import spacy
import pandas as pd

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Actual columns expected in the raw CSV file
REQUIRED_COLUMNS = [
    "course_title",
    "description",
    "level",
    "subject",
]

# Path to dataset (relative to project root)
DATASET_PATH = "data/Courses_dataset.csv"

# Column mapping: raw CSV columns → internal standard schema
COLUMN_MAPPING = {
    "course_title": "course_name",
    "description": "description",
    "level":        "difficulty",
    "subject":      "department",
}


# -------------------------------
# DATA LOADING
# -------------------------------
def load_dataset(path=DATASET_PATH):
    """
    Load the course dataset from CSV.
    - Validates required columns exist in raw file
    - Renames columns to internal standard schema
    - Adds constant 'university' column (Udemy)
    Returns a standardized DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Validate required raw columns exist before renaming
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"[WARNING] Missing expected columns: {missing_cols}")
        else:
            print(f"[INFO] All required columns present.")

        # Rename to internal standard schema
        df = df.rename(columns=COLUMN_MAPPING)

        # Add constant university column (scalable: override per-row if future
        # datasets include a university field)
        if "university" not in df.columns:
            df["university"] = "Udemy"

        print(f"[INFO] Columns standardized to internal schema.")
        return df

    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at path: {path}")
        raise


# -------------------------------
# DATA CLEANING
# -------------------------------
def clean_dataset(df):
    """
    Clean the standardized DataFrame:
    - Drop rows with missing course_name or description
    - Remove duplicate entries based on course_name + description
    - Strip leading/trailing whitespace from string columns
    - Reset index after cleaning
    Returns a cleaned DataFrame.
    """
    original_count = len(df)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Drop rows missing critical fields (using internal schema names)
    df = df.dropna(subset=["course_name", "description"])

    # Remove duplicates based on title + description (using internal schema names)
    df = df.drop_duplicates(subset=["course_name", "description"])

    # Reset index cleanly
    df = df.reset_index(drop=True)

    cleaned_count = len(df)
    print(f"[INFO] Cleaning complete: {original_count} → {cleaned_count} rows "
          f"({original_count - cleaned_count} removed)")

    return df


# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    """
    Clean a raw text string:
    - Handle NaN/None values
    - Remove HTML tags
    - Remove special characters and punctuation
    - Normalize whitespace
    - Convert to lowercase
    """
    if pd.isna(text):
        return ""

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', str(text))

    # Remove special characters and punctuation (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Normalize multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


# -------------------------------
# FULL PREPROCESSING PIPELINE
# -------------------------------
def preprocess_text(text):
    """
    Full NLP preprocessing pipeline for a single text string:
    - Clean raw text
    - Tokenize using spaCy
    - Remove stopwords and punctuation
    - Apply lemmatization
    - Keep tokens with 2+ characters
    Returns a single preprocessed string.
    """
    text = clean_text(text)

    if not text:
        return ""

    doc = nlp(text)

    # Lemmatization + Stopword Removal
    # Keeping tokens >= 2 chars (relaxed from >2 to capture valid short terms)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) >= 2
    ]

    return " ".join(tokens)


# -------------------------------
# DATASET-LEVEL PREPROCESSING
# -------------------------------
def preprocess_dataset(df, text_column="description"):
    """
    Apply the full preprocessing pipeline to an entire DataFrame column.
    Adds 'processed_description' column with cleaned, lemmatized text.
    Guarantees output always contains: course_name, description, processed_description.
    Scalable for large datasets (8500+ rows).
    Returns the DataFrame with the new column added.
    """
    print(f"[INFO] Preprocessing '{text_column}' column for {len(df)} courses...")

    df = df.copy()

    # Apply preprocessing pipeline — stored as 'processed_description'
    df["processed_description"] = df[text_column].apply(preprocess_text)

    # Report how many rows ended up empty after preprocessing
    empty_count = (df["processed_description"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"[WARNING] {empty_count} rows have empty text after preprocessing.")

    print(f"[INFO] Preprocessing complete.")

    # Guarantee key columns are present in output
    for col in ["course_name", "description", "processed_description"]:
        if col not in df.columns:
            print(f"[WARNING] Expected column '{col}' not found in output DataFrame.")

    return df