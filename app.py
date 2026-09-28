"""
FAQ Chatbot - Flask Backend
Matches user questions to the closest FAQ using TF-IDF + Cosine Similarity.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# One-time NLTK data download (safe to run every start, it skips if present)
# ---------------------------------------------------------------------------
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("faqs.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(tokens)


df["processed_question"] = df["question"].apply(preprocess)

# ---------------------------------------------------------------------------
# Build TF-IDF matrix once at startup
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["processed_question"])

CONFIDENCE_THRESHOLD = 0.2


def get_best_answer(user_question: str):
    processed = preprocess(user_question)
    if not processed.strip():
        return "Please type a valid question.", 0.0

    user_vector = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = similarities.argmax()
    best_score = float(similarities[best_idx])

    if best_score < CONFIDENCE_THRESHOLD:
        return (
            "Sorry, I couldn't find a good match for that. "
            "Try rephrasing, or contact support@example.com.",
            best_score,
        )
    return df.iloc[best_idx]["answer"], best_score


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json(force=True)
    user_message = (data or {}).get("message", "").strip()

    if not user_message:
        return jsonify({"answer": "Please type a question.", "confidence": 0})

    answer, score = get_best_answer(user_message)
    return jsonify({"answer": answer, "confidence": round(score, 2)})


if __name__ == "__main__":
    app.run(debug=True)
