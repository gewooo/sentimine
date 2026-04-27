from fastapi import FastAPI, UploadFile, File, HTTPException  # Core FastAPI objects for building routes, handling uploads, and signaling HTTP errors.
from fastapi.middleware.cors import CORSMiddleware  # Middleware that adds CORS headers so the frontend can call this API from another origin.
from pydantic import BaseModel  # Base class for request/response schemas with validation.
import pandas as pd  # Pandas is used to read CSV files and manipulate tabular review data.
import numpy as np  # NumPy supports numeric operations and array handling used by the metrics code.
from sklearn.model_selection import train_test_split  # Splits the dataset into training and test partitions.
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts raw review text into TF-IDF features.
from sklearn.linear_model import LogisticRegression  # Linear classifier used as one of the baseline models.
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier that works well with sparse text features.
from sklearn.svm import LinearSVC  # Linear support vector machine used for classification.
from sklearn.calibration import CalibratedClassifierCV  # Wraps LinearSVC so it can expose probability-like outputs.
from sklearn.ensemble import RandomForestClassifier  # Tree ensemble classifier used as another baseline.
from sklearn.metrics import accuracy_score, classification_report  # Metrics used to evaluate each trained model.
import io  # In-memory bytes buffer utilities for reading uploaded CSV files.
from typing import Optional  # Optional type helper imported for future nullable annotations.
import warnings  # Lets the code suppress noisy library warnings during training and inference.
import os  # File-system helpers used to search for the training dataset in multiple locations.
import pickle  # Used to save the fixed best model and vectorizer for reuse.
import re  # Used for lightweight tokenization in explanations and word clouds.
from collections import Counter  # Used to build word frequency data for dashboard responses.
warnings.filterwarnings("ignore")  # Hide warnings so model training output stays readable in the console.

app = FastAPI(title="SentimentIQ Dashboard API")  # Create the FastAPI application instance and give it a descriptive title.

app.add_middleware(  # Register middleware that runs around every request.
    CORSMiddleware,  # Use the standard CORS middleware for browser-based frontend calls.
    allow_origins=["*"],  # Allow requests from any origin during development.
    allow_methods=["*"],  # Allow every HTTP method, including GET, POST, and OPTIONS.
    allow_headers=["*"],  # Allow every request header so uploads and browser requests do not get blocked.
)

# Global state keeps the loaded dataset, trained models, and computed metrics in memory for the lifetime of the server.
state = {
    "df": None,  # The currently loaded pandas DataFrame, or None before any dataset has been loaded.
    "vectorizer": None,  # The fitted TF-IDF vectorizer used to transform review text into features.
    "models": {},  # A dictionary of trained sentiment and emotion models keyed by algorithm name.
    "metrics": {},  # Training and evaluation metrics for each algorithm pair.
    "is_trained": False,  # A flag that tells the API whether model training has completed successfully.
    "dataset_stats": {},  # Cached dataset summary values returned by the stats endpoints.
    "best_model": "SVM",  # Best model by weighted F1-score after training.
    "text_col": None,  # Active review text column after schema normalization.
    "dataset_name": "None",  # Name of the currently active dataset for dashboard display.
}

ALGORITHM_NAMES = ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"]  # Friendly labels for the supported algorithm families.
TEXT_COLUMNS = ["Customer Review", "Customer Review (English)", "review"]  # Supported review-text column names.
EMOTION_MAP = {"Happy": "happy", "Sadness": "sad", "Sad": "sad", "Anger": "angry", "Angry": "angry"}
EMOTION_KEYS = ["happy", "sad", "angry"]
# modelArtifactPaths: Store all trained models and the TF-IDF vectorizer in the backend directory.
BACKEND_DIR = os.path.dirname(__file__)
MODEL_PATHS = {
    "SVM": os.path.join(BACKEND_DIR, "svm_model.pkl"),
    "Logistic Regression": os.path.join(BACKEND_DIR, "logistic_regression_model.pkl"),
    "Naive Bayes": os.path.join(BACKEND_DIR, "naive_bayes_model.pkl"),
    "Random Forest": os.path.join(BACKEND_DIR, "random_forest_model.pkl"),
}
VECTORIZER_PATH = os.path.join(BACKEND_DIR, "vectorizer.pkl")
# Legacy fallback path
MODEL_ARTIFACT_PATH = os.path.join(BACKEND_DIR, "best_model.pkl")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "is", "it", "as", "for",
    "be", "by", "from", "with", "are", "was", "were", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "can", "if", "that", "this",
    "what", "which", "who", "when", "where", "why", "how", "all", "each", "every", "both", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "not", "no", "so", "very",
}
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "like", "delicious", "perfect",
    "happy", "fast", "safe", "thank", "thanks", "nice", "best", "quality", "trusted", "recommend",
    "yummy", "tasty", "flavorful", "fresh", "satisfied", "awesome", "wonderful", "pleased", "beautiful",
}
SAD_WORDS = {
    "sad", "disappointed", "disappointing", "poor", "broken", "late", "missing", "wrong", "damaged",
    "bad", "waste", "slow", "issue", "problem", "terrible", "unhappy", "return",
}
ANGRY_WORDS = {
    "angry", "mad", "hate", "hated", "rude", "scam", "fake", "awful", "worst", "annoying",
    "frustrated", "unhelpful", "refund", "complaint", "damn", "useless",
}


def read_dataset_csv(path: str) -> pd.DataFrame:  # Read a CSV that may use UTF-8 or legacy spreadsheet encodings.
    try:
        return pd.read_csv(path)  # Prefer UTF-8/default pandas behavior first.
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")  # filtered_dataset.csv contains bytes that need this fallback.


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:  # Resolve a logical column from several possible names.
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def ensure_training_schema(df: pd.DataFrame, require_emotion: bool = True) -> tuple[pd.DataFrame, str, str, str]:  # Normalize the dataset columns used by training.
    text_col = find_column(df, TEXT_COLUMNS)
    emotion_col = find_column(df, ["Emotion", "emotion"])
    category_col = find_column(df, ["Category", "category"])
    product_col = find_column(df, ["Product Name", "product name", "product"])

    missing = []
    if text_col is None:
        missing.append("Customer Review")
    if require_emotion and emotion_col is None:
        missing.append("Emotion")
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    normalized = df.copy()
    if emotion_col is not None and emotion_col != "Emotion":
        normalized["Emotion"] = normalized[emotion_col]
    
    if require_emotion and "Sentiment" not in normalized.columns and "Emotion" in normalized.columns:
        normalized["Sentiment"] = normalized["Emotion"].map(lambda emotion: "Positive" if emotion == "Happy" else "Negative")

    return normalized, text_col, category_col, product_col


def normalize_emotion(value: str) -> str:
    return EMOTION_MAP.get(str(value), EMOTION_MAP.get(str(value).title(), "happy"))


def display_emotion(value: str) -> str:
    normalized = normalize_emotion(value)
    return {"happy": "happy", "sad": "sad", "angry": "angry"}[normalized]


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z']+", str(text).lower()) if len(w) > 2 and w not in STOP_WORDS]


def build_probabilities(model, X) -> dict:
    raw = {"happy": 0.0, "sad": 0.0, "angry": 0.0}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        for label, prob in zip(model.classes_, probs):
            raw[normalize_emotion(label)] = round(float(prob), 4)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)[0], dtype=float)
        exp = np.exp(scores - np.max(scores))
        probs = exp / exp.sum()
        for label, prob in zip(model.classes_, probs):
            raw[normalize_emotion(label)] = round(float(prob), 4)
    return raw


def best_model_name() -> str:
    if state["best_model"] in state["models"]:
        return state["best_model"]
    if state["models"]:
        # Order of preference if best_model is not available
        for name in ALGORITHM_NAMES:
            if name in state["models"]:
                return name
        return next(iter(state["models"].keys()))
    return "SVM"


def summarize_dataset(df: pd.DataFrame, use_existing_emotions: bool = True, algorithm: Optional[str] = None) -> dict:
    df, text_col, category_col, product_col = ensure_training_schema(df, require_emotion=False)
    working = df.copy()

    # Identify which model to use for inference - strictly prioritize the requested algorithm
    model_used = algorithm if algorithm in state["models"] else best_model_name()
    
    # We force re-prediction if an algorithm is selected, or if the user explicitly asks for inference,
    # or if the dataset is unlabeled (missing "Emotion" column).
    should_repredict = not use_existing_emotions or algorithm is not None or "Emotion" not in working.columns
    
    if should_repredict:
        # Optimized batch prediction to avoid row-by-row overhead and bypass heuristics for comparison
        if model_used in state["models"]:
            model = state["models"][model_used]["emotion"]
            X = state["vectorizer"].transform(working[text_col].fillna("").astype(str))
            preds = model.predict(X)
            # Use raw model predictions mapped to our standard keys (happy, sad, angry)
            working["Emotion"] = [normalize_emotion(p) for p in preds]
        else:
            # Fallback to single prediction if models are not loaded (should be rare)
            working["Emotion"] = [predict_emotion(str(text), algorithm=model_used)["emotion"] for text in working[text_col].fillna("")]

    working["emotion_key"] = working["Emotion"].map(normalize_emotion)
    counts = working["emotion_key"].value_counts().to_dict()

    def grouped(col):
        if not col or col not in working.columns:
            return []
        rows = []
        top_names = working[col].fillna("Unknown").value_counts().head(10).index
        for name in top_names:
            subset = working[working[col].fillna("Unknown") == name]
            emo_counts = subset["emotion_key"].value_counts().to_dict()
            rows.append({
                "name": str(name),
                "happy": int(emo_counts.get("happy", 0)),
                "sad": int(emo_counts.get("sad", 0)),
                "angry": int(emo_counts.get("angry", 0)),
            })
        return rows

    word_counts = {"all": Counter(), "happy": Counter(), "sad": Counter(), "angry": Counter()}
    for _, row in working.iterrows():
        emotion = normalize_emotion(row.get("Emotion", "happy"))
        words = tokenize(row.get(text_col, ""))
        word_counts["all"].update(words)
        word_counts[emotion].update(words)

    wordcloud = []
    for emotion in ["happy", "sad", "angry"]:
        for word, count in word_counts[emotion].most_common(28):
            wordcloud.append({"word": word, "count": int(count), "emotion": emotion})

    # Category vs Emotion Heatmap Data - Limited to Top 10 categories by frequency
    cat_emo_matrix = {}
    working_cat_col = category_col if category_col in working.columns else "Category"
    if working_cat_col not in working.columns:
        working[working_cat_col] = "Unknown"
    
    # Get top 10 categories to keep the heatmap clean
    top_cats = working[working_cat_col].fillna("Unknown").value_counts().head(10).index
    subset_working = working[working[working_cat_col].isin(top_cats)]
    
    pivot = pd.crosstab(subset_working[working_cat_col], subset_working["emotion_key"])
    # Reorder pivot by the top_cats order
    pivot = pivot.reindex(top_cats)
    cat_emo_matrix = pivot.to_dict(orient="index")

    # Review Length analysis removed
    return {
        # "results": Select only necessary columns (Category, Product, Review text, Emotion), keep first 200 rows, convert to JSON-friendly records for frontend table display.
        "results": working[[c for c in [category_col, product_col, text_col, "Emotion"] if c and c in working.columns]].head(200).to_dict(orient="records"),
        # "summary": Build KPI counters showing total reviews and emotion distribution (happy/sad/angry breakdown).
        "summary": {
            "total": int(len(working)),  # Total number of reviews in the dataset after filtering.
            "happy_count": int(counts.get("happy", 0)),  # Count of reviews classified as happy/positive emotion.
            "sad_count": int(counts.get("sad", 0)),  # Count of reviews classified as sad/disappointing emotion.
            "angry_count": int(counts.get("angry", 0)),  # Count of reviews classified as angry/frustrated emotion.
        },
        # "top_products": Group reviews by product name and count emotions for each top-10 product (used for dashboard product breakdown chart).
        "top_products": grouped(product_col),
        # "top_categories": Group reviews by category and count emotions for each top-10 category (used for dashboard category breakdown chart).
        "top_categories": grouped(category_col),
        # "wordcloud": Word frequency data showing most common words per emotion (used for word cloud visualization on dashboard).
        "wordcloud": wordcloud,
        "category_emotion_matrix": cat_emo_matrix,
        "algorithm_used": model_used,
        "dataset_name": state.get("dataset_name", "Unknown"),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def normalize_classification_dataset(df: pd.DataFrame) -> pd.DataFrame:
    text_col = find_column(df, TEXT_COLUMNS)
    if text_col is None:
        raise ValueError("Missing required column: Customer Review")
    normalized = df.copy()
    if "Emotion" not in normalized.columns:
        normalized["Emotion"] = "Happy"
    if "Sentiment" not in normalized.columns:
        normalized["Sentiment"] = "Positive"
    return normalized


def predict_emotion(sentence: str, algorithm: Optional[str] = None) -> dict:
    if not state["is_trained"]:
        raise HTTPException(status_code=400, detail="No model trained yet.")

    model_name = algorithm if algorithm in state["models"] else best_model_name()
    model = state["models"][model_name]["emotion"]
    X = state["vectorizer"].transform([sentence])
    predicted = model.predict(X)[0]
    probabilities = build_probabilities(model, X)
    words = []
    dominant = normalize_emotion(predicted)
    raw_words = re.findall(r"[a-zA-Z][a-zA-Z']*", sentence)
    lowered_words = [word.lower() for word in raw_words]
    positive_hits = [word for word in lowered_words if word in POSITIVE_WORDS]
    sad_hits = [word for word in lowered_words if word in SAD_WORDS]
    angry_hits = [word for word in lowered_words if word in ANGRY_WORDS]

    if positive_hits and not sad_hits and not angry_hits:
        dominant = "happy"
        probabilities = {
            "happy": max(probabilities.get("happy", 0), 0.82),
            "sad": min(probabilities.get("sad", 0), 0.12),
            "angry": min(probabilities.get("angry", 0), 0.12),
        }
    elif angry_hits and not positive_hits:
        dominant = "angry"
        probabilities = {
            "happy": min(probabilities.get("happy", 0), 0.12),
            "sad": min(probabilities.get("sad", 0), 0.18),
            "angry": max(probabilities.get("angry", 0), 0.82),
        }
    elif sad_hits and not positive_hits:
        dominant = "sad"
        probabilities = {
            "happy": min(probabilities.get("happy", 0), 0.12),
            "sad": max(probabilities.get("sad", 0), 0.82),
            "angry": min(probabilities.get("angry", 0), 0.18),
        }

    has_known_signal = any(
        word in POSITIVE_WORDS or word in SAD_WORDS or word in ANGRY_WORDS
        for word in lowered_words
    )

    for word in raw_words:
        clean_word = word.lower()
        base = max(probabilities.get(dominant, 0.33), 0.12)
        scores = {
            "happy": round(probabilities.get("happy", 0) * 0.12, 4),
            "sad": round(probabilities.get("sad", 0) * 0.12, 4),
            "angry": round(probabilities.get("angry", 0) * 0.12, 4),
        }
        if clean_word in POSITIVE_WORDS:
            scores["happy"] = round(max(base, 0.95), 4)
        elif clean_word in SAD_WORDS:
            scores["sad"] = round(max(base, 0.95), 4)
        elif clean_word in ANGRY_WORDS:
            scores["angry"] = round(max(base, 0.95), 4)
        elif not has_known_signal and clean_word not in STOP_WORDS and len(clean_word) > 5:
            scores[dominant] = round(base * 0.9, 4)
        else:
            scores[dominant] = round(base * 0.18, 4)
        words.append({"word": word, "scores": scores})

    return {"emotion": dominant, "probabilities": probabilities, "words": words}


@app.on_event("startup")  # Run this coroutine once when the API server starts.
async def startup_event():  # Load the default dataset and train models automatically at boot.
    """Load and train models on datasets/filtered_dataset.csv on startup."""  # Short docstring describing the startup behavior.
    try:  # Wrap startup work so a missing dataset does not crash the whole server.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # originalDatasetToModelTraining: Load datasets/filtered_dataset.csv as the training source.
        # Define absolute paths for datasets based on the known workspace structure
        dataset_dir = os.path.join(project_root, "datasets")
        labeled_path = os.path.join(dataset_dir, "filtered_dataset.csv")
        unlabeled_path = os.path.join(dataset_dir, "unlabeled_dataset.csv")

        df = None
        if os.path.exists(labeled_path):
            df = read_dataset_csv(labeled_path)
            print(f"Loaded training dataset from {labeled_path}")
        else:
            print(f"Warning: Training dataset not found at {labeled_path}")

        if df is not None:
            # trainModelFromOriginalDataset: Train the candidate models using the original filtered dataset.
            train_all_models(df)
            
            # Switch to unlabeled_dataset.csv for the dashboard default display
            if os.path.exists(unlabeled_path):
                unlabeled_df = read_dataset_csv(unlabeled_path)
                state["df"] = ensure_training_schema(unlabeled_df, require_emotion=False)[0]
                state["dataset_name"] = "unlabeled_dataset.csv"
                # Initialize dataset_stats using inference on the unlabeled data
                initial_summary = summarize_dataset(state["df"], use_existing_emotions=False)
                state["dataset_stats"] = initial_summary.get("summary", {})
                print(f"Switched dashboard to unlabeled dataset: {unlabeled_path}")
            else:
                state["df"] = ensure_training_schema(df, require_emotion=False)[0]
                state["dataset_name"] = "filtered_dataset.csv"
                print(f"Unlabeled dataset missing at {unlabeled_path}, using labeled one for display.")
        else:
            print("No training dataset found. Training deferred.")
    except Exception as e:  # Catch any unexpected startup failure so the server still boots.
        print(f"Error loading default dataset: {e}")  # Log the exception message for debugging.
        import traceback  # Import traceback only when a startup error happens.
        traceback.print_exc()  # Print the full stack trace to the console.


def train_all_models(df: pd.DataFrame):  # Train every supported model pair on the provided dataset.
    # modelTrainingPipeline: Clean labels, vectorize reviews, train models, compare F1, and save the best model.
    df, text_col, category_col, product_col = ensure_training_schema(df)  # Support filtered_dataset.csv and legacy CSV schemas.
    state["text_col"] = text_col
    cat_counts_full = df[category_col].value_counts().head(10).to_dict() if category_col else {}  # Count common categories when available.
    prod_counts_full = df[product_col].value_counts().head(10).to_dict() if product_col else {}  # Count common products when available.

    ALLOWED_EMOTIONS = ["Happy", "Sadness", "Anger"]  # Keep only the emotion labels that the dashboard is designed to show.
    df = df[df["Emotion"].isin(ALLOWED_EMOTIONS)]  # Remove rows whose emotion is outside the supported set.
    df = df.dropna(subset=[text_col, "Sentiment", "Emotion"])  # Drop rows missing the key text or label columns.

    print(f"Emotion distribution: {df['Emotion'].value_counts().to_dict()}")  # Log the filtered emotion balance for visibility.
    print(f"Training on {len(df)} reviews with {len(df['Emotion'].unique())} emotions")  # Log how many rows and classes remain after filtering.

    X = df[text_col].astype(str)  # Convert review text to strings so TF-IDF can process every row safely.
    y_sent = df["Sentiment"]  # Sentiment labels used for the sentiment classifier.
    y_emo = df["Emotion"]  # Emotion labels used for the emotion classifier.

    X_train, X_test, ys_train, ys_test, ye_train, ye_test = train_test_split(  # Split once so both tasks use the same training and test rows.
        X, y_sent, y_emo, test_size=0.2, random_state=42  # Use an 80/20 split with a fixed seed for repeatable results.
    )

    # tfidfVectorizerTraining: Turn review text into numeric features for model training.
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Create a TF-IDF vectorizer with up to 5000 uni-grams and bi-grams.
    X_tr = vec.fit_transform(X_train)  # Learn the vocabulary from the training reviews and convert them to sparse features.
    X_te = vec.transform(X_test)  # Convert the held-out reviews with the same learned vocabulary.

    model_defs = {  # Map each algorithm name to its separate sentiment and emotion estimators.
        "Logistic Regression": (  # Strong linear baseline that usually performs well on sparse text.
            LogisticRegression(max_iter=1000, class_weight='balanced'),  # Sentiment model with class balancing for imbalance.
            LogisticRegression(max_iter=1000, class_weight='balanced'),  # Emotion model with the same configuration.
        ),
        "Naive Bayes": (  # Simple probabilistic baseline that is fast and works well on text counts.
            MultinomialNB(),  # Sentiment model.
            MultinomialNB(),  # Emotion model.
        ),
        "SVM": (  # Linear SVM wrapped in calibration so confidence scores can be produced later.
            CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')),  # Sentiment model.
            CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')),  # Emotion model.
        ),
        "Random Forest": (  # Tree ensemble baseline that gives a contrasting model family.
            RandomForestClassifier(n_estimators=60, n_jobs=1, random_state=42, class_weight='balanced'),  # Sentiment model.
            RandomForestClassifier(n_estimators=60, n_jobs=1, random_state=42, class_weight='balanced'),  # Emotion model.
        ),
    }

    trained = {}  # Store fitted model objects here.
    metrics = {}  # Store per-algorithm evaluation metrics here.

    # trainCandidateModels: Train Logistic Regression, Naive Bayes, SVM, and Random Forest.
    for name, (ms, me) in model_defs.items():  # Train and evaluate each algorithm pair.
        try:
            ms.fit(X_tr, ys_train)  # Fit the sentiment classifier on the TF-IDF training matrix.
            ps = ms.predict(X_te)  # Predict sentiment labels on the test matrix.
            sa = accuracy_score(ys_test, ps)  # Compute sentiment accuracy.
            sr = classification_report(ys_test, ps, output_dict=True)  # Generate a detailed sentiment metric report.

            me.fit(X_tr, ye_train)  # Fit the emotion classifier on the same training matrix.
            pe = me.predict(X_te)  # Predict emotion labels on the test matrix.
            ea = accuracy_score(ye_test, pe)  # Compute emotion accuracy.
            er = classification_report(ye_test, pe, output_dict=True)  # Generate a detailed emotion metric report.

            print(f"  {name}: Sentiment={sa:.4f}, Emotion={ea:.4f}")  # Print a short summary line for the console.

            trained[name] = {"sentiment": ms, "emotion": me}  # Keep the fitted sentiment and emotion models for later prediction.
            metrics[name] = {  # Save the evaluation results for this algorithm.
                "sentiment": {  # Metrics for the sentiment classifier.
                    "accuracy": round(sa, 4),  # Store rounded sentiment accuracy.
                    "report": sr,  # Store the full classification report dictionary.
                },
                "emotion": {  # Metrics for the emotion classifier.
                    "accuracy": round(ea, 4),  # Store rounded emotion accuracy.
                    "report": er,  # Store the full classification report dictionary.
                },
            }
        except Exception as e:
            print(f"  {name}: skipped ({e})")

    state["vectorizer"] = vec  # Save the fitted TF-IDF vectorizer for later inference.
    state["models"] = trained  # Save every trained algorithm pair in memory.
    state["metrics"] = metrics  # Save the collected evaluation metrics in memory.
    # chooseBestModelByF1Score: Pick the model with the highest weighted emotion F1-score.
    state["best_model"] = max(
        metrics,
        key=lambda name: metrics[name]["emotion"]["report"].get("weighted avg", {}).get("f1-score", 0),
    )
    state["is_trained"] = True  # Mark the system as ready for prediction requests.
    try:
        # saveAllModelsPklArtifacts: Save every trained model and vectorizer to their respective .pkl files.
        for name, paths in state["models"].items():
            if name in MODEL_PATHS:
                with open(MODEL_PATHS[name], "wb") as f:
                    pickle.dump(paths["emotion"], f)
        
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(state["vectorizer"], f)

        # saveBestModelPklArtifact: Keep legacy best_model.pkl for backward compatibility.
        with open(MODEL_ARTIFACT_PATH, "wb") as artifact:
            pickle.dump({
                "best_model": state["best_model"],
                "vectorizer": state["vectorizer"],
                "model": state["models"][state["best_model"]]["emotion"],
            }, artifact)
    except Exception as e:
        print(f"Could not save model artifacts: {e}")

    emo_counts = df["Emotion"].value_counts().to_dict()  # Count the filtered emotion labels for the dataset summary.
    sent_counts = df["Sentiment"].value_counts().to_dict()  # Count the sentiment labels for the dataset summary.

    state["dataset_stats"] = {  # Build the summary object returned by the stats endpoints.
        "total_reviews": int(len(df)),  # Record how many rows survived the filtering step.
        "emotion_distribution": {k: int(v) for k, v in emo_counts.items()},  # Convert emotion counts to plain Python ints.
        "sentiment_distribution": {k: int(v) for k, v in sent_counts.items()},  # Convert sentiment counts to plain Python ints.
        "category_distribution": {k: int(v) for k, v in cat_counts_full.items()},  # Keep the top category counts from the full dataset.
        "product_distribution": {k: int(v) for k, v in prod_counts_full.items()},  # Keep the top product counts from the full dataset.
        "avg_rating": round(float(df["Overall Rating"].mean()), 2) if "Overall Rating" in df.columns else None,  # Compute rating when present.
        "columns": df.columns.tolist(),  # Preserve the dataset column names for the frontend.
        "categories": df[category_col].unique().tolist() if category_col else [],  # Preserve the unique categories present in the filtered data.
    }

    return metrics  # Return the metrics dictionary in case the caller wants to inspect it immediately.


@app.get("/")  # Root endpoint used as a simple health/status check.
def root():  # Return a minimal JSON response confirming the API is running.
    return {"status": "SentimentIQ API running", "trained": state["is_trained"]}  # Tell the caller whether models are ready.


class ClassifyRequest(BaseModel):
    sentence: str
    algorithm: Optional[str] = None


@app.post("/api/classify")
def api_classify(req: ClassifyRequest):
    # useTrainedModelForHeroClassification: Run the Home page sentence through the trained in-memory model.
    if not req.sentence.strip():
        raise HTTPException(status_code=400, detail="Sentence is required.")
    return predict_emotion(req.sentence.strip(), algorithm=req.algorithm)


@app.get("/api/dataset/original")
def api_original_dataset(algorithm: Optional[str] = None):
    # fetchOriginalDatasetDashboardData: Return KPI/chart data from the original filtered dataset.
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="Original dataset is not loaded.")
    # Force inference mode (use_existing_emotions=False) so the dashboard 
    # reflects the selected model's predictions on the unlabeled dataset.
    return summarize_dataset(state["df"], use_existing_emotions=False, algorithm=algorithm)


@app.post("/api/classify-dataset")
async def api_classify_dataset(file: UploadFile = File(...), algorithm: Optional[str] = None):
    # classifyUploadedDatasetWithSelectedModel: Classify uploaded dataset rows with the chosen model.
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")
        normalized = normalize_classification_dataset(df)
        return summarize_dataset(normalized, use_existing_emotions=False, algorithm=algorithm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/comparison")
def api_models_comparison():
    # fetchModelComparisonMetrics: Return model metrics for the Home page comparison chart.
    if not state["metrics"]:
        raise HTTPException(status_code=400, detail="Models are not trained yet.")

    models = []
    for name, metric in state["metrics"].items():
        weighted = metric["emotion"]["report"].get("weighted avg", {})
        models.append({
            "name": name,
            "accuracy": float(metric["emotion"].get("accuracy", 0)),
            "precision": float(weighted.get("precision", 0)),
            "recall": float(weighted.get("recall", 0)),
            "f1": float(weighted.get("f1-score", 0)),
        })

    best = max(models, key=lambda item: item["f1"])["name"] if models else state["best_model"]
    state["best_model"] = best
    return {"models": models, "best_model": best}


@app.post("/upload-dataset")  # Endpoint that accepts a new CSV and retrains the models.
async def upload_dataset(file: UploadFile = File(...)):  # Read the uploaded file from a multipart form request.
    try:  # Handle file parsing and retraining failures cleanly.
        content = await file.read()  # Read the entire upload into memory as bytes.
        try:
            df = pd.read_csv(io.BytesIO(content))  # Parse the bytes into a pandas DataFrame.
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")  # Accept CSVs exported with legacy encodings.

        ensure_training_schema(df)  # Validate the uploaded training schema and support the filtered dataset columns.

        state["df"] = ensure_training_schema(df)[0]  # Replace the in-memory dataset with a normalized uploaded one.
        metrics = train_all_models(df)  # Retrain all supported algorithms on the new dataset.

        return {  # Return upload and retraining details to the frontend.
            "success": True,  # Signal that the upload and retraining completed successfully.
            "rows": int(len(df)),  # Report how many rows were uploaded.
            "columns": df.columns.tolist(),  # Echo the uploaded column names.
            "metrics": metrics,  # Include the fresh metrics from the retraining run.
            "dataset_stats": state["dataset_stats"],  # Include the refreshed dataset summary.
        }
    except HTTPException:  # Allow intentionally raised HTTP errors to pass through unchanged.
        raise
    except Exception as e:  # Convert any other failure into a 500 response.
        raise HTTPException(status_code=500, detail=str(e))  # Return the exception message for debugging.


class PredictRequest(BaseModel):  # Request schema for the /predict endpoint.
    text: str  # The review text to classify.
    algorithm: str = "SVM"  # The model family to use, defaulting to SVM.


@app.post("/predict")  # Endpoint that predicts sentiment and emotion for a single text sample.
def predict(req: PredictRequest):  # Read the validated request body and run inference.
    if not state["is_trained"]:  # Refuse to predict until training has happened.
        raise HTTPException(status_code=400, detail="No model trained. Upload a dataset first.")  # Tell the client what to do next.

    algo = req.algorithm  # Read the algorithm requested by the client.
    if algo not in state["models"]:  # Verify that the requested algorithm exists in memory.
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algo}")  # Reject invalid algorithm names.

    vec = state["vectorizer"]  # Reuse the fitted TF-IDF vectorizer from training.
    X = vec.transform([req.text])  # Transform the single input text into a sparse feature row.

    sent_pred = state["models"][algo]["sentiment"].predict(X)[0]  # Predict the sentiment label using the chosen algorithm.
    emo_pred = state["models"][algo]["emotion"].predict(X)[0]  # Predict the emotion label using the chosen algorithm.

    sent_proba = {}  # Store sentiment confidence values when the model supports them.
    emo_proba = {}  # Store emotion confidence values when the model supports them.

    sm = state["models"][algo]["sentiment"]  # Alias the sentiment model to keep the code below readable.
    em = state["models"][algo]["emotion"]  # Alias the emotion model to keep the code below readable.

    if hasattr(sm, "predict_proba"):  # If the sentiment model can output probabilities...
        proba = sm.predict_proba(X)[0]  # Get the probability vector for the single sample.
        sent_proba = dict(zip(sm.classes_, [round(float(p), 4) for p in proba]))  # Map each class label to its rounded probability.
    elif hasattr(sm, "decision_function"):  # Otherwise, use decision scores if the model exposes them.
        scores = sm.decision_function(X)[0]  # Compute raw decision scores.
        if scores.ndim == 0:  # Handle the binary-classifier case where only one scalar score is returned.
            classes = sm.classes_  # Read the class labels from the classifier.
            sent_proba = {classes[0]: round(float(-scores), 4), classes[1]: round(float(scores), 4)}  # Convert the score into two class-aligned values.
        else:  # Handle the multi-class case where an array of scores is returned.
            sent_proba = dict(zip(sm.classes_, [round(float(s), 4) for s in scores]))  # Pair each class with its score.

    if hasattr(em, "predict_proba"):  # If the emotion model can output probabilities...
        proba = em.predict_proba(X)[0]  # Get the probability vector for the sample.
        emo_proba = dict(zip(em.classes_, [round(float(p), 4) for p in proba]))  # Map each emotion class to its rounded probability.
    elif hasattr(em, "decision_function"):  # Otherwise, fall back to decision scores.
        scores = em.decision_function(X)[0]  # Compute raw emotion scores.
        emo_proba = dict(zip(em.classes_, [round(float(s), 4) for s in scores]))  # Pair each emotion class with its score.

    all_results = {}  # Collect predictions from every trained algorithm for comparison.
    for name, mods in state["models"].items():  # Iterate over each algorithm pair that was trained.
        s = mods["sentiment"].predict(X)[0]  # Predict sentiment with this specific algorithm.
        e = mods["emotion"].predict(X)[0]  # Predict emotion with this specific algorithm.
        algo_emo_conf = None  # Default to no emotion confidence data for this algorithm.
        if hasattr(mods["emotion"], "predict_proba"):  # If the emotion model supports probabilities...
            proba_e = mods["emotion"].predict_proba(X)[0]  # Get the probability vector for the emotion prediction.
            algo_emo_conf = {c: round(float(p), 4) for c, p in zip(mods["emotion"].classes_, proba_e)}  # Map each emotion class to its probability.
        all_results[name] = {"sentiment": s, "emotion": e, "emotion_confidence": algo_emo_conf}  # Store the comparison entry.

    return {  # Return the main prediction plus the comparison table.
        "text": req.text,  # Echo the input text back to the client.
        "algorithm": algo,  # Echo the algorithm that was requested.
        "sentiment": sent_pred,  # Main sentiment prediction.
        "emotion": emo_pred,  # Main emotion prediction.
        "sentiment_confidence": sent_proba,  # Confidence or score map for sentiment.
        "emotion_confidence": emo_proba,  # Confidence or score map for emotion.
        "all_algorithms": all_results,  # Per-algorithm comparison results.
    }


@app.get("/metrics")  # Endpoint that returns all saved training metrics.
def get_metrics():  # Return the metrics object if the models have already been trained.
    if not state["is_trained"]:  # Refuse the request if the training step has not happened yet.
        raise HTTPException(status_code=400, detail="No model trained.")  # Tell the client that training is missing.
    return {"metrics": state["metrics"], "dataset_stats": state["dataset_stats"]}  # Return both metrics and dataset summary.


@app.get("/dataset-stats")  # Endpoint that returns only the cached dataset summary.
def get_stats():  # Provide the dataset summary without the full metrics payload.
    if not state["dataset_stats"]:  # Fail if no dataset has been loaded yet.
        raise HTTPException(status_code=400, detail="No dataset loaded.")  # Tell the client to upload data first.
    return state["dataset_stats"]  # Return the cached summary object.


@app.get("/sample-reviews")  # Endpoint that returns a random sample of reviews for previewing the dataset.
def sample_reviews(n: int = 10):  # Accept an optional sample size, defaulting to 10.
    if state["df"] is None:  # Refuse to sample if there is no dataset in memory.
        raise HTTPException(status_code=400, detail="No dataset loaded.")  # Tell the client that nothing is available to sample.
    df, text_col, category_col, _ = ensure_training_schema(state["df"])  # Support both filtered and legacy dataset schemas.
    df = df.dropna(subset=[text_col, "Sentiment", "Emotion"])  # Remove rows missing the key fields before sampling.
    sample_columns = [col for col in [category_col, text_col, "Sentiment", "Emotion", "Customer Rating", "Overall Rating"] if col and col in df.columns]
    sample = df[sample_columns].sample(min(n, len(df))).to_dict(orient="records")  # Pick a random subset and convert it to JSON-friendly records.
    return {"reviews": sample}  # Return the sampled rows.


if __name__ == "__main__":  # Only execute the server launch code when this file is run directly.
    import uvicorn  # Import the ASGI server runner only for the direct-run path.
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Start the FastAPI app on port 8000 for local development.
