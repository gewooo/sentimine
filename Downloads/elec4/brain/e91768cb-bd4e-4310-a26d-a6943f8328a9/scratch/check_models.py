import pandas as pd
import pickle
import os

backend_dir = r"c:\Users\Jnorlynne\Downloads\elec4\backend"
models = {
    "SVM": os.path.join(backend_dir, "svm_model.pkl"),
    "Naive Bayes": os.path.join(backend_dir, "naive_bayes_model.pkl"),
    "Logistic Regression": os.path.join(backend_dir, "logistic_regression_model.pkl"),
    "Random Forest": os.path.join(backend_dir, "random_forest_model.pkl"),
}
vec_path = os.path.join(backend_dir, "vectorizer.pkl")
data_path = r"c:\Users\Jnorlynne\Downloads\elec4\datasets\unlabeled_dataset.csv"

with open(vec_path, "rb") as f:
    vec = pickle.load(f)

df = pd.read_csv(data_path, encoding="latin1")
text_col = "Customer Review"
X = vec.transform(df[text_col].fillna("").astype(str))

results = {}
for name, path in models.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        preds = model.predict(X)
        counts = pd.Series(preds).value_counts().to_dict()
        results[name] = counts
    else:
        results[name] = "MISSING"

print(results)
