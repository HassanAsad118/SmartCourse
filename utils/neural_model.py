import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import NEURAL_MODEL_PATH as MODEL_PATH

from utils.preprocessing import (
    load_dataset,
    clean_dataset,
    preprocess_dataset,
    preprocess_text,
    clean_text,
)


# Load pre-trained semantic model (loaded once at module level for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# QUERY EXPANSION MAP
# -------------------------------
QUERY_EXPANSION_MAP = {
    "python":           "python programming coding scripting",
    "data science":     "data science machine learning data analysis statistics",
    "machine learning": "machine learning deep learning neural network AI artificial intelligence",
    "web":              "web development html css javascript frontend backend",
    "excel":            "excel spreadsheet data analysis business",
    "sql":              "sql database query data management",
    "finance":          "finance financial accounting investment banking",
    "marketing":        "marketing digital marketing seo social media",
    "design":           "design graphic ui ux creative visual",
    "business":         "business management strategy entrepreneurship",
    "cloud":            "cloud computing aws azure devops infrastructure",
    "security":         "security cybersecurity network ethical hacking",
    "android":          "android mobile app development java kotlin",
    "ios":              "ios mobile app development swift apple",
    "photography":      "photography camera editing lightroom",
    "music":            "music audio production instrument theory",
    "health":           "health fitness nutrition wellness medical",
    "language":         "language english communication writing speaking",
    "drawing":          "drawing art illustration sketching",
    "leadership":       "leadership management team communication soft skills",
}


# -------------------------------
# QUERY EXPANSION FUNCTION
# -------------------------------
def expand_query(query):
    """
    Expand a cleaned query string with semantically related terms.
    """
    expanded_terms = [query]

    for keyword, expansion in QUERY_EXPANSION_MAP.items():
        if keyword in query:
            expanded_terms.append(expansion)

    return " ".join(expanded_terms)


# -------------------------------
# TRAIN NEURAL EMBEDDING MODEL
# -------------------------------
def train_neural():
    """
    Full training pipeline for Neural Embedding model:
    - Loads and cleans dataset using preprocessing pipeline
    - Applies NLP preprocessing to descriptions
    - Generates sentence embeddings using all-MiniLM-L6-v2
    - Saves embeddings and cleaned DataFrame to disk
    """
    print("[INFO] Loading dataset...")
    df = load_dataset()

    print("[INFO] Cleaning dataset...")
    df = clean_dataset(df)

    print("[INFO] Preprocessing descriptions...")
    df = preprocess_dataset(df)

    print("[INFO] Generating embeddings using all-MiniLM-L6-v2...")
    print("[INFO] This may take a few minutes for large datasets...")

    embeddings = model.encode(
        df["processed_description"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True
    )

    joblib.dump((embeddings, df), MODEL_PATH)
    print(f"[INFO] Neural embeddings saved to '{MODEL_PATH}' successfully.")


# -------------------------------
# RECOMMEND USING NEURAL MODEL
# -------------------------------
def recommend_neural(query, top_k=10):
    """
    Generate top-k course recommendations using Neural embeddings.

    Query handling pipeline:
    1. clean_text()      — remove HTML, punctuation, normalize whitespace, lowercase
    2. preprocess_text() — lemmatization + stopword removal (same as training)
    3. expand_query()    — append semantically related terms for richer context
    4. model.encode()    — convert enriched query into embedding space

    Returns ranked list with title, department, description, score, and url.
    """
    embeddings, df = joblib.load(MODEL_PATH)

    # Step 1: Clean raw query
    query_cleaned = clean_text(query)

    # Step 2: Preprocess query
    query_preprocessed = preprocess_text(query_cleaned)

    # Step 3: Expand query
    query_expanded = expand_query(query_preprocessed)

    # Step 4: Encode enriched query
    query_embedding = model.encode(
        [query_expanded],
        convert_to_numpy=True
    )

    # Compute cosine similarity
    similarity = cosine_similarity(query_embedding, embeddings).flatten()

    # Get top-k indices sorted by descending similarity
    top_indices = similarity.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = float(similarity[idx])
        score_percent = round(score * 100, 2) if score > 0 else 0.0

        results.append({
            "title":       df.iloc[idx]["course_name"],
            "department":  df.iloc[idx]["department"],
            "description": df.iloc[idx]["description"],
            "score":       score_percent,
            # url column passed through for frontend "View Course" button
            "url":         df.iloc[idx]["url"] if "url" in df.columns else "#",
        })

    return results


# -------------------------------
# EVALUATION METRICS
# -------------------------------
def evaluate_neural(queries, relevant_courses_per_query, k=10):
    """
    Evaluate Neural model using standard IR metrics.
    Mirrors evaluate_tfidf() exactly for fair side-by-side comparison.
    """
    precision_scores = []
    recall_scores    = []
    hit_rates        = []
    per_query_results = []

    for query, relevant_set in zip(queries, relevant_courses_per_query):
        recommendations = recommend_neural(query, top_k=k)
        recommended_titles = {r["title"] for r in recommendations}

        true_positives = recommended_titles & relevant_set

        precision = len(true_positives) / k
        recall    = len(true_positives) / len(relevant_set) if relevant_set else 0.0
        hit       = 1.0 if len(true_positives) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_rates.append(hit)

        per_query_results.append({
            "query":     query,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "hit":       hit,
        })

        print(f"[EVAL] Query: '{query}' | "
              f"Precision@{k}: {precision:.4f} | "
              f"Recall@{k}: {recall:.4f} | "
              f"Hit: {hit}")

    avg_precision = round(sum(precision_scores) / len(precision_scores), 4)
    avg_recall    = round(sum(recall_scores)    / len(recall_scores),    4)
    avg_hit_rate  = round(sum(hit_rates)        / len(hit_rates),        4)

    print(f"\n[EVAL] Average Precision@{k}: {avg_precision}")
    print(f"[EVAL] Average Recall@{k}:    {avg_recall}")
    print(f"[EVAL] Average Hit-Rate@{k}:  {avg_hit_rate}")

    return {
        "avg_precision": avg_precision,
        "avg_recall":    avg_recall,
        "avg_hit_rate":  avg_hit_rate,
        "per_query":     per_query_results,
    }