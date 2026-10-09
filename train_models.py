import time
from utils.tfidf_model import train_tfidf
from utils.neural_model import train_neural


# -----------------------------
# MAIN TRAINING PIPELINE
# -----------------------------
def main():
    print("\n" + "=" * 55)
    print("  SmartCourse — Full Model Training Pipeline")
    print("=" * 55)
    print("[INFO] Starting full model training...\n")

    total_start = time.time()

    # --- Step 1: TF-IDF Model ---
    print("[INFO] Training TF-IDF model...")
    tfidf_start = time.time()
    train_tfidf()
    tfidf_time = round(time.time() - tfidf_start, 2)
    print(f"[SUCCESS] TF-IDF model trained in {tfidf_time}s\n")

    # --- Step 2: Neural Embedding Model ---
    print("[INFO] Training Neural model...")
    neural_start = time.time()
    train_neural()
    neural_time = round(time.time() - neural_start, 2)
    print(f"[SUCCESS] Neural model trained in {neural_time}s\n")

    # --- Summary ---
    total_time = round(time.time() - total_start, 2)
    print("=" * 55)
    print(f"[INFO] All models trained successfully in {total_time}s")
    print("=" * 55 + "\n")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()