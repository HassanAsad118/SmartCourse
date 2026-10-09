from utils.tfidf_model import recommend_tfidf
from utils.neural_model import recommend_neural

# -------------------------------
# TEST QUERIES + RELEVANT KEYWORDS
# -------------------------------
# Each entry defines:
# - query:    the user search string
# - keywords: terms expected to appear in relevant course titles/descriptions
#             A recommendation is considered relevant if ANY keyword matches.
TEST_QUERIES = [
    {
        "query": "python for data science",
        "keywords": ["python", "data", "machine learning", "analysis", "science"]
    },
    {
        "query": "web development with javascript",
        "keywords": ["web", "javascript", "html", "css", "frontend", "development"]
    },
    {
        "query": "machine learning and deep learning",
        "keywords": ["machine learning", "deep learning", "neural", "ai", "tensorflow"]
    },
    {
        "query": "business finance and accounting",
        "keywords": ["finance", "accounting", "business", "financial", "investment"]
    },
    {
        "query": "graphic design and photoshop",
        "keywords": ["design", "photoshop", "graphic", "illustrator", "creative"]
    },
    {
        "query": "sql and database management",
        "keywords": ["sql", "database", "query", "mysql", "data management"]
    },
    {
        "query": "digital marketing and seo",
        "keywords": ["marketing", "seo", "digital", "social media", "advertising"]
    },
    {
        "query": "android mobile app development",
        "keywords": ["android", "mobile", "app", "java", "kotlin", "development"]
    },
    {
        "query": "excel for business and data analysis",
        "keywords": ["excel", "spreadsheet", "business", "data", "analysis"]
    },
    {
        "query": "leadership and project management",
        "keywords": ["leadership", "management", "project", "team", "agile"]
    },
]


# -------------------------------
# RELEVANCE CHECK
# -------------------------------
def is_relevant(result, keywords):
    """
    Check if a single recommendation result is relevant to a query.
    Relevance is determined by keyword presence in:
    - course title (lower-cased)
    - course description (lower-cased)

    A result is relevant if ANY of the keywords appear in either field.
    This approach is practical when exact course labels are unavailable.
    """
    title       = result.get("title", "").lower()
    description = result.get("description", "").lower()

    for keyword in keywords:
        if keyword.lower() in title or keyword.lower() in description:
            return True

    return False


# -------------------------------
# SINGLE MODEL EVALUATOR
# -------------------------------
def evaluate_model(model_name, recommend_fn, k=10):
    """
    Evaluate a recommendation model using keyword-based relevance.

    Parameters:
    - model_name:    display name for output (e.g. "TF-IDF", "Neural")
    - recommend_fn:  callable that accepts (query, top_k) and returns results
    - k:             number of top results to evaluate (default: 10)

    For each test query:
    - Fetches top-k recommendations
    - Checks each result for keyword relevance
    - Computes Precision@k, Recall@k, Hit-Rate@k

    Metrics:
    - Precision@k:  fraction of top-k results that are relevant
    - Recall@k:     fraction of all keywords found in top-k results
                    (approximated as: unique keywords matched / total keywords)
    - Hit-Rate@k:   1.0 if at least one relevant result in top-k, else 0.0

    Returns aggregated averages across all test queries.
    """
    precision_scores = []
    recall_scores    = []
    hit_rates        = []

    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*55}")

    for entry in TEST_QUERIES:
        query    = entry["query"]
        keywords = entry["keywords"]

        # Get top-k recommendations from model
        results = recommend_fn(query, top_k=k)

        # Count how many results are relevant (keyword match)
        relevant_results = [r for r in results if is_relevant(r, keywords)]
        num_relevant     = len(relevant_results)

        # Precision@k: relevant results / k
        precision = num_relevant / k

        # Recall@k: approximated as fraction of keywords found
        # across all top-k result titles and descriptions
        matched_keywords = set()
        for result in results:
            title       = result.get("title", "").lower()
            description = result.get("description", "").lower()
            for keyword in keywords:
                if keyword.lower() in title or keyword.lower() in description:
                    matched_keywords.add(keyword.lower())

        recall = len(matched_keywords) / len(keywords) if keywords else 0.0

        # Hit-Rate@k: 1 if at least one relevant result found
        hit = 1.0 if num_relevant > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_rates.append(hit)

        print(f"\n  Query     : {query}")
        print(f"  Relevant  : {num_relevant}/{k} results matched keywords")
        print(f"  Precision@{k}: {precision:.4f}")
        print(f"  Recall@{k}   : {recall:.4f}")
        print(f"  Hit-Rate  : {hit:.1f}")

    # Aggregate averages
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall    = sum(recall_scores)    / len(recall_scores)
    avg_hit_rate  = sum(hit_rates)        / len(hit_rates)

    print(f"\n{'-'*55}")
    print(f"  {model_name} — Final Averages (k={k})")
    print(f"{'-'*55}")
    print(f"  Precision@{k} : {avg_precision:.4f}")
    print(f"  Recall@{k}    : {avg_recall:.4f}")
    print(f"  Hit-Rate@{k}  : {avg_hit_rate:.4f}")
    print(f"{'-'*55}")

    return {
        "model":         model_name,
        "avg_precision": round(avg_precision, 4),
        "avg_recall":    round(avg_recall,    4),
        "avg_hit_rate":  round(avg_hit_rate,  4),
    }


# -------------------------------
# COMPARE BOTH MODELS
# -------------------------------
def run_evaluation(k=10):
    """
    Run full evaluation for both TF-IDF and Neural models.
    Prints side-by-side comparison of all metrics.
    Returns results dict for both models.
    """
    tfidf_results  = evaluate_model("TF-IDF",  recommend_tfidf,  k=k)
    neural_results = evaluate_model("Neural",  recommend_neural, k=k)

    # Side-by-side summary
    print(f"\n{'='*55}")
    print(f"  FINAL COMPARISON SUMMARY (k={k})")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'TF-IDF':>10} {'Neural':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Precision@'+str(k):<20} "
          f"{tfidf_results['avg_precision']:>10.4f} "
          f"{neural_results['avg_precision']:>10.4f}")
    print(f"  {'Recall@'+str(k):<20} "
          f"{tfidf_results['avg_recall']:>10.4f} "
          f"{neural_results['avg_recall']:>10.4f}")
    print(f"  {'Hit-Rate@'+str(k):<20} "
          f"{tfidf_results['avg_hit_rate']:>10.4f} "
          f"{neural_results['avg_hit_rate']:>10.4f}")
    print(f"{'='*55}\n")

    return {
        "tfidf":  tfidf_results,
        "neural": neural_results,
    }


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_evaluation(k=10)