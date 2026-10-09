import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TFIDF_MODEL_PATH as MODEL_PATH

from utils.preprocessing import (
    load_dataset,
    clean_dataset,
    preprocess_dataset,
    preprocess_text,
)


# -------------------------------
# TRAIN TF-IDF MODEL
# -------------------------------
def train_tfidf():
    """
    Full training pipeline for TF-IDF model:
    - Loads and cleans dataset using preprocessing pipeline
    - Applies NLP preprocessing to descriptions
    - Trains TF-IDF vectorizer with unigram + bigram (1,2)
    - Saves vectorizer, TF-IDF matrix, and cleaned DataFrame to disk
    """
    print("[INFO] Loading dataset...")
    df = load_dataset()

    print("[INFO] Cleaning dataset...")
    df = clean_dataset(df)

    print("[INFO] Preprocessing descriptions...")
    df = preprocess_dataset(df)

    print("[INFO] Training TF-IDF vectorizer (ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["processed_description"])

    # Save vectorizer, matrix, and processed DataFrame together
    joblib.dump((vectorizer, tfidf_matrix, df), MODEL_PATH)
    print(f"[INFO] TF-IDF model saved to '{MODEL_PATH}' successfully.")


# -------------------------------
# RECOMMEND USING TF-IDF
# -------------------------------
def recommend_tfidf(query, top_k=10):
    """
    Generate top-k course recommendations for a user query using TF-IDF.
    - Preprocesses the query using the same pipeline as training
    - Computes cosine similarity between query and all course vectors
    - Returns ranked list with title, department, description, score, and url
    """
    vectorizer, tfidf_matrix, df = joblib.load(MODEL_PATH)

    # Preprocess query using same pipeline as training
    query_processed = preprocess_text(query)
    query_vec = vectorizer.transform([query_processed])

    # Compute cosine similarity between query and all courses
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top-k indices sorted by descending similarity
    top_indices = similarity.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = float(similarity[idx])

        # Normalize score to 0-100%, guard against zero scores
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
def evaluate_tfidf(queries, relevant_courses_per_query, k=10):
    """
    Evaluate TF-IDF model using standard IR metrics.

    Parameters:
    - queries: list of query strings
        e.g. ["python for data science", "machine learning basics"]
    - relevant_courses_per_query: list of sets, each containing relevant
        course titles for the corresponding query
        e.g. [{"Python Bootcamp", "Data Science A-Z"}, {"ML Course"}]
    - k: number of top results to evaluate against (default: 10)

    Metrics computed:
    - Precision@k: fraction of top-k results that are relevant
    - Recall@k:    fraction of relevant courses found in top-k results
    - Hit-Rate@k:  fraction of queries where at least 1 relevant result
                   appears in top-k results

    Returns a dict with per-query breakdown and aggregate averages.
    """
    precision_scores = []
    recall_scores    = []
    hit_rates        = []

    per_query_results = []

    for query, relevant_set in zip(queries, relevant_courses_per_query):
        recommendations = recommend_tfidf(query, top_k=k)
        recommended_titles = {r["title"] for r in recommendations}

        # True positives: recommended courses that are relevant
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

    # Aggregate averages across all queries
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