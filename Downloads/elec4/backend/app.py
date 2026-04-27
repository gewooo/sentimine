from flask import Flask, request, jsonify  # Flask core pieces used to define routes, read requests, and return JSON.
from flask_cors import CORS  # Adds CORS headers so the frontend can talk to this backend from another origin.
import pandas as pd  # Pandas is used to load, clean, filter, and summarize the review CSV.
import numpy as np  # NumPy is used for class handling and matrix utilities during evaluation.
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts review text into TF-IDF features.
from sklearn.linear_model import LogisticRegression  # Logistic regression model used for classification.
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes model used for sparse text features.
from sklearn.svm import SVC  # Support vector classifier used as the SVM option.
from sklearn.ensemble import RandomForestClassifier  # Random forest model used as another baseline.
from sklearn.model_selection import train_test_split, cross_val_score  # train_test_split is used for holdout evaluation.
from sklearn.metrics import (  # Metrics used to evaluate each trained model.
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder  # Encodes emotion labels into integers and back.
import warnings  # Used to silence noisy library warnings during training.
import json  # Imported for JSON handling if needed by future additions.
import re  # Regular expressions are used to clean review text.
import os  # Used to resolve the canonical dataset path from the repository root.

warnings.filterwarnings('ignore')  # Hide warnings so the training logs stay readable.

app = Flask(__name__)  # Create the Flask application instance.
CORS(app)  # Allow browser-based frontend requests to reach this API.

# Global state stores the loaded dataset, fitted vectorizers, trained models, and metrics in memory.
df = None  # The current dataset loaded into memory.
models = {}  # Trained sentiment/emotion models keyed by algorithm name.
vectorizer_sentiment = None  # TF-IDF vectorizer for sentiment classification.
vectorizer_emotion = None  # TF-IDF vectorizer for emotion classification.
le_emotion = None  # Label encoder used to map emotion strings to integers.
model_metrics = {}  # Cached metrics for the dashboard and API responses.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Repository root, used for stable file lookup.
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "filtered_dataset.csv")  # Canonical training/original dataset.
TEXT_COLUMNS = ["Customer Review", "Customer Review (English)", "review"]  # Supported review-text column names.


def read_dataset_csv(source):  # Read CSV files that may use UTF-8 or legacy spreadsheet encodings.
    try:
        return pd.read_csv(source)  # Prefer pandas' default UTF-8 path first.
    except UnicodeDecodeError:
        return pd.read_csv(source, encoding="latin1")  # filtered_dataset.csv needs this fallback on some machines.


def find_column(dataframe, candidates):  # Resolve a logical column from several supported spellings.
    lowered = {col.lower(): col for col in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def ensure_training_schema(dataframe):  # Normalize the dataset columns needed by the model training code.
    text_col = find_column(dataframe, TEXT_COLUMNS)
    emotion_col = find_column(dataframe, ["Emotion", "emotion"])
    category_col = find_column(dataframe, ["Category", "category"])
    product_col = find_column(dataframe, ["Product Name", "product name", "product"])

    missing = []
    if text_col is None:
        missing.append("Customer Review")
    if emotion_col is None:
        missing.append("Emotion")
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    normalized = dataframe.copy()
    if emotion_col != "Emotion":
        normalized["Emotion"] = normalized[emotion_col]
    if "Sentiment" not in normalized.columns:
        normalized["Sentiment"] = normalized["Emotion"].map(lambda emotion: "Positive" if emotion == "Happy" else "Negative")

    return normalized, text_col, category_col, product_col


def preprocess_text(text):  # Normalize one review so it can be turned into model features.
    if not isinstance(text, str):  # If the input is missing or not text, treat it as empty.
        return ""  # Return an empty string so vectorization does not break.
    text = text.lower()  # Lowercase everything so word matching is case-insensitive.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove punctuation and symbols, keep letters, numbers, and spaces.
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse repeated whitespace and trim the result.
    return text  # Return the cleaned review text.


def train_models(dataframe):  # Train sentiment and emotion models from a DataFrame.
    global models, vectorizer_sentiment, vectorizer_emotion, le_emotion, model_metrics  # Update module-level caches.

    dataframe, text_col, _, _ = ensure_training_schema(dataframe)  # Support filtered_dataset.csv and legacy CSV schemas.
    df_clean = dataframe.dropna(subset=[text_col, "Sentiment", "Emotion"])  # Remove rows missing required training fields.
    df_clean = df_clean.copy()  # Work on a copy so later edits do not affect the original frame.
    # Only train on the 3 emotions shown in the dashboard.
    df_clean = df_clean[df_clean["Emotion"].isin(["Happy", "Sadness", "Anger"])]  # Keep only the supported emotion labels.
    df_clean["processed"] = df_clean[text_col].apply(preprocess_text)  # Clean every review before feature extraction.
    df_clean = df_clean[df_clean["processed"].str.len() > 2]  # Remove reviews that became too short after cleaning.

    X_text = df_clean["processed"].values  # Final cleaned review text used as model input.
    y_sentiment = df_clean["Sentiment"].values  # Target labels for sentiment prediction.
    y_emotion = df_clean["Emotion"].values  # Target labels for emotion prediction.

    # Encode emotion labels.
    le_emotion = LabelEncoder()  # Create a label encoder for emotion strings.
    y_emotion_encoded = le_emotion.fit_transform(y_emotion)  # Convert emotion names into integer classes.

    # TF-IDF vectorizers.
    vectorizer_sentiment = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Build the sentiment feature extractor.
    X_sent = vectorizer_sentiment.fit_transform(X_text)  # Learn the sentiment vocabulary and transform all text.

    vectorizer_emotion = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Build the emotion feature extractor.
    X_emo = vectorizer_emotion.fit_transform(X_text)  # Learn the emotion vocabulary and transform all text.

    # Train-test split.
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(  # Split sentiment features and labels into train/test sets.
        X_sent, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment  # Stratify so class proportions stay balanced.
    )
    X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(  # Split emotion features and labels into train/test sets.
        X_emo, y_emotion_encoded, test_size=0.2, random_state=42, stratify=y_emotion_encoded  # Stratify on encoded emotion labels.
    )

    algorithm_defs = {  # Define the sentiment and emotion model pair for each algorithm family.
        "logistic_regression": {  # Logistic regression baseline.
            "sentiment": LogisticRegression(max_iter=500, random_state=42),  # Sentiment model.
            "emotion": LogisticRegression(max_iter=500, random_state=42),  # Emotion model.
        },
        "naive_bayes": {  # Multinomial Naive Bayes baseline.
            "sentiment": MultinomialNB(),  # Sentiment model.
            "emotion": MultinomialNB(),  # Emotion model.
        },
        "svm": {  # Linear SVM classifier.
            "sentiment": SVC(kernel="linear", probability=True, max_iter=1000, random_state=42),  # Sentiment model with probabilities enabled.
            "emotion": SVC(kernel="linear", probability=True, max_iter=1000, random_state=42),  # Emotion model with probabilities enabled.
        },
        "random_forest": {  # Random forest baseline.
            "sentiment": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),  # Sentiment model.
            "emotion": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),  # Emotion model.
        },
    }

    models = {}  # Reset the model cache before storing the new trained models.
    model_metrics = {}  # Reset the metrics cache before storing fresh evaluation results.

    for algo_key, algo_models in algorithm_defs.items():  # Train and evaluate each algorithm pair.
        # --- Sentiment ---
        sm = algo_models["sentiment"]  # Get the sentiment classifier for this algorithm.
        sm.fit(X_s_train, y_s_train)  # Train it on the sentiment training data.
        s_pred = sm.predict(X_s_test)  # Predict sentiment on the test split.

        s_acc = accuracy_score(y_s_test, s_pred)  # Compute sentiment accuracy.
        s_prec = precision_score(y_s_test, s_pred, average="weighted", zero_division=0)  # Compute weighted precision.
        s_rec = recall_score(y_s_test, s_pred, average="weighted", zero_division=0)  # Compute weighted recall.
        s_f1 = f1_score(y_s_test, s_pred, average="weighted", zero_division=0)  # Compute weighted F1.

        s_classes = list(np.unique(y_s_test))  # Collect the sentiment classes present in the test split.
        s_cm = confusion_matrix(y_s_test, s_pred, labels=s_classes).tolist()  # Build a JSON-friendly confusion matrix.

        # --- Emotion ---
        em = algo_models["emotion"]  # Get the emotion classifier for this algorithm.
        em.fit(X_e_train, y_e_train)  # Train it on the emotion training data.
        e_pred = em.predict(X_e_test)  # Predict emotion on the test split.

        e_acc = accuracy_score(y_e_test, e_pred)  # Compute emotion accuracy.
        e_prec = precision_score(y_e_test, e_pred, average="weighted", zero_division=0)  # Compute weighted emotion precision.
        e_rec = recall_score(y_e_test, e_pred, average="weighted", zero_division=0)  # Compute weighted emotion recall.
        e_f1 = f1_score(y_e_test, e_pred, average="weighted", zero_division=0)  # Compute weighted emotion F1.

        e_classes_encoded = list(np.unique(y_e_test))  # Collect the encoded emotion classes in the test split.
        e_classes = le_emotion.inverse_transform(e_classes_encoded).tolist()  # Convert encoded classes back to readable labels.
        e_cm = confusion_matrix(y_e_test, e_pred, labels=e_classes_encoded).tolist()  # Build the emotion confusion matrix.

        models[algo_key] = {"sentiment": sm, "emotion": em}  # Store the fitted models for later prediction requests.
        model_metrics[algo_key] = {  # Store the metrics for dashboard display.
            "sentiment": {  # Sentiment metric block.
                "accuracy": round(s_acc, 4),  # Rounded sentiment accuracy.
                "precision": round(s_prec, 4),  # Rounded sentiment precision.
                "recall": round(s_rec, 4),  # Rounded sentiment recall.
                "f1": round(s_f1, 4),  # Rounded sentiment F1.
                "confusion_matrix": s_cm,  # Sentiment confusion matrix.
                "classes": s_classes,  # Sentiment class labels.
            },
            "emotion": {  # Emotion metric block.
                "accuracy": round(e_acc, 4),  # Rounded emotion accuracy.
                "precision": round(e_prec, 4),  # Rounded emotion precision.
                "recall": round(e_rec, 4),  # Rounded emotion recall.
                "f1": round(e_f1, 4),  # Rounded emotion F1.
                "confusion_matrix": e_cm,  # Emotion confusion matrix.
                "classes": e_classes,  # Emotion class labels.
            },
        }

    return True  # Return True to signal that training finished successfully.


def load_default_dataset():  # Load the default CSV from disk and train models immediately.
    global df  # This function updates the module-level dataset reference.
    try:  # Catch missing-file and parsing errors so startup can continue gracefully.
        df = ensure_training_schema(read_dataset_csv(DATASET_PATH))[0]  # Read and normalize the default dataset into a DataFrame.
        train_models(df)  # Train all models on the loaded dataset.
        print(f"[BOOT] Loaded {len(df)} rows. Models trained.")  # Log the boot-time success message.
    except Exception as e:  # Handle any startup failure.
        print(f"[BOOT] Failed: {e}")  # Log the error instead of crashing the server.


# Routes define the HTTP API used by the frontend.

@app.route("/api/health")  # Lightweight status endpoint for checking if the backend is alive.
def health():  # Return a short summary of runtime readiness.
    return jsonify({"status": "ok", "models_ready": bool(models), "rows": len(df) if df is not None else 0})  # Report service state and row count.


@app.route("/api/upload", methods=["POST"])  # Endpoint that accepts a new CSV and retrains the models.
def upload_csv():  # Read a multipart upload, validate it, and replace the in-memory dataset.
    global df  # This route overwrites the module-level dataset.
    if "file" not in request.files:  # Reject requests that do not include a file field.
        return jsonify({"error": "No file provided"}), 400  # Tell the client what is missing.
    file = request.files["file"]  # Read the uploaded file object from the request.
    if not file.filename.endswith(".csv"):  # Require a CSV file extension.
        return jsonify({"error": "Only CSV files are supported"}), 400  # Reject non-CSV uploads.
    try:  # Wrap parsing and training so upload failures become a clean JSON response.
        new_df = read_dataset_csv(file)  # Parse the uploaded CSV into a DataFrame.
        ensure_training_schema(new_df)  # Validate that the uploaded file can train the emotion model.
        df = ensure_training_schema(new_df)[0]  # Replace the old dataset with the newly uploaded and normalized one.
        train_models(df)  # Retrain every model on the new dataset.
        return jsonify({"success": True, "rows": len(df), "message": "Dataset loaded & models retrained"})  # Confirm the upload and retraining completed.
    except Exception as e:  # Catch CSV parsing and training failures.
        return jsonify({"error": str(e)}), 500  # Return the exception as a server error message.


@app.route("/api/stats")  # Endpoint that summarizes the currently loaded dataset.
def get_stats():  # Compute and return counts and cross-tabs used by the dashboard.
    if df is None:  # Refuse the request if no dataset has been loaded yet.
        return jsonify({"error": "No dataset loaded"}), 400  # Tell the client to upload or load data first.

    sentiment_counts = df["Sentiment"].value_counts().to_dict() if "Sentiment" in df.columns else {}  # Count available sentiment labels.
    emotion_counts = df["Emotion"].value_counts().to_dict()  # Count how many rows belong to each emotion label.
    category_counts = df["Category"].value_counts().head(10).to_dict() if "Category" in df.columns else {}  # Keep top categories when available.

    avg_rating = round(df["Customer Rating"].mean(), 2) if "Customer Rating" in df.columns else None  # Compute the average rating when the column exists.

    # Sentiment x Emotion cross-tab shows how the two label dimensions overlap.
    if "Sentiment" in df.columns:  # Only build this table when sentiment labels exist or were derived.
        cross = df.groupby(["Sentiment", "Emotion"]).size().reset_index(name="count")  # Aggregate counts for each sentiment/emotion pair.
        cross_data = cross.to_dict(orient="records")  # Convert the grouped table into JSON-friendly records.
    else:
        cross_data = []

    # Category sentiment breakdown shows sentiment counts within each product category.
    if "Category" in df.columns and "Sentiment" in df.columns:  # Only compute this breakdown when Category and Sentiment exist.
        cat_sent = (  # Build the grouped category/sentiment table.
            df.groupby(["Category", "Sentiment"])  # Group rows by category and sentiment.
            .size()  # Count how many rows fall into each group.
            .reset_index(name="count")  # Convert the grouped Series into a DataFrame with a count column.
            .to_dict(orient="records")  # Convert to a JSON-friendly list of dictionaries.
        )
    else:  # If Category is missing, return an empty list instead of failing.
        cat_sent = []  # No category-level breakdown is available.

    return jsonify({  # Return the computed dataset summary to the frontend.
        "total_reviews": len(df),  # Total number of loaded rows.
        "sentiment_counts": sentiment_counts,  # Sentiment distribution across the dataset.
        "emotion_counts": emotion_counts,  # Emotion distribution across the dataset.
        "category_counts": category_counts,  # Top category counts.
        "avg_rating": avg_rating,  # Average customer rating when available.
        "cross_tab": cross_data,  # Sentiment/emotion cross-tab records.
        "category_sentiment": cat_sent,  # Category/sentiment breakdown records.
    })


@app.route("/api/metrics")  # Endpoint that returns the training metrics for each algorithm.
def get_metrics():  # Send the cached evaluation metrics to the client.
    if not model_metrics:  # Refuse the request if the models have not been trained yet.
        return jsonify({"error": "Models not trained"}), 400  # Tell the client to load or upload data first.
    return jsonify(model_metrics)  # Return the full metrics dictionary.


@app.route("/api/analyze", methods=["POST"])  # Endpoint that predicts sentiment and emotion for one review.
def analyze():  # Read the request text, clean it, vectorize it, and run every model.
    if not models:  # Refuse prediction if training has not happened yet.
        return jsonify({"error": "Models not trained yet"}), 400  # Tell the client to load data first.

    data = request.get_json()  # Parse the incoming JSON body from the frontend.
    text = data.get("text", "").strip()  # Extract the review text and trim surrounding whitespace.
    algorithm = data.get("algorithm", "logistic_regression")  # Pick the requested model family, defaulting to logistic regression.

    if not text:  # Stop if the user sent an empty review.
        return jsonify({"error": "No text provided"}), 400  # Explain that text input is required.
    if algorithm not in models:  # Stop if the requested algorithm does not exist in the trained cache.
        return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400  # Report the invalid algorithm name.

    processed = preprocess_text(text)  # Apply the same text cleaning used during training.

    # Sentiment prediction uses the TF-IDF vectorizer fitted on training data.
    X_s = vectorizer_sentiment.transform([processed])  # Convert the cleaned review into sentiment features.
    sentiment_pred = models[algorithm]["sentiment"].predict(X_s)[0]  # Predict the sentiment label.

    # Sentiment confidence is collected when the classifier supports probabilities.
    sentiment_conf = None  # Default value when probabilities are unavailable.
    try:  # Some models may not expose predict_proba, so this is optional.
        proba_s = models[algorithm]["sentiment"].predict_proba(X_s)[0]  # Get class probabilities for the sentiment prediction.
        classes_s = models[algorithm]["sentiment"].classes_  # Read the sentiment class labels from the model.
        sentiment_conf = {c: round(float(p), 4) for c, p in zip(classes_s, proba_s)}  # Build a label-to-probability map.
    except Exception:  # Fall back silently if the model cannot provide probabilities.
        pass

    # Emotion prediction uses the separate emotion vectorizer and classifier.
    X_e = vectorizer_emotion.transform([processed])  # Convert the cleaned review into emotion features.
    emotion_pred_encoded = models[algorithm]["emotion"].predict(X_e)[0]  # Predict the encoded emotion class.
    emotion_pred = le_emotion.inverse_transform([emotion_pred_encoded])[0]  # Convert the encoded class back to a readable emotion.

    # Emotion confidence is collected when the classifier supports probabilities.
    emotion_conf = None  # Default value when probabilities are unavailable.
    try:  # This branch only works for probability-capable classifiers.
        proba_e = models[algorithm]["emotion"].predict_proba(X_e)[0]  # Get emotion class probabilities.
        classes_e = le_emotion.inverse_transform(models[algorithm]["emotion"].classes_)  # Convert encoded classes back to emotion names.
        emotion_conf = {c: round(float(p), 4) for c, p in zip(classes_e, proba_e)}  # Build a label-to-probability map.
    except Exception:  # Ignore classifiers that do not expose probabilities.
        pass

    # All algorithms comparison lets the frontend show how every model voted on the same text.
    comparison = {}  # Container for per-algorithm comparison results.
    for algo_key, algo_model in models.items():  # Evaluate each trained algorithm pair on the same input.
        s_pred = algo_model["sentiment"].predict(X_s)[0]  # Predict sentiment with this model pair.
        e_enc = algo_model["emotion"].predict(X_e)[0]  # Predict encoded emotion with this model pair.
        e_pred = le_emotion.inverse_transform([e_enc])[0]  # Convert the encoded emotion back to its string label.
        # Per-algorithm emotion confidence is optional and only available for some estimators.
        algo_emo_conf = None  # Default when probabilities are not supported.
        try:  # Attempt to get per-class emotion probabilities.
            proba_ae = algo_model["emotion"].predict_proba(X_e)[0]  # Get probability values for the emotion classes.
            classes_ae = le_emotion.inverse_transform(algo_model["emotion"].classes_)  # Convert encoded class names back to strings.
            algo_emo_conf = {c: round(float(p), 4) for c, p in zip(classes_ae, proba_ae)}  # Build the probability map.
        except Exception:  # If the model does not support probabilities, leave this field empty.
            pass
        comparison[algo_key] = {  # Store the prediction summary for this algorithm.
            "sentiment": s_pred,  # Sentiment predicted by this model.
            "emotion": e_pred,  # Emotion predicted by this model.
            "emotion_confidence": algo_emo_conf,  # Optional emotion probability map.
        }

    return jsonify({  # Return the main prediction plus the comparison table.
        "text": text,  # Echo the input text back to the client.
        "algorithm": algorithm,  # Echo the selected algorithm.
        "sentiment": sentiment_pred,  # Primary sentiment result.
        "sentiment_confidence": sentiment_conf,  # Sentiment probabilities when available.
        "emotion": emotion_pred,  # Primary emotion result.
        "emotion_confidence": emotion_conf,  # Emotion probabilities when available.
        "all_algorithms": comparison,  # Side-by-side predictions from every model.
    })


if __name__ == "__main__":  # Only run the server when this file is executed directly.
    load_default_dataset()  # Load the default CSV and train models before the server starts.
    app.run(debug=False, port=5000)  # Start the Flask development server on port 5000.
