import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path



csv_path = Path(__file__).parent / "spam.csv"

df = pd.read_csv(csv_path, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


pipe = Pipeline([
    ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), min_df=2)),
    ("nb", MultinomialNB(alpha=0.5))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
print("Relatório de classificação:\n", classification_report(y_test, y_pred, digits=3))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))

