# FAQ Chatbot (Flask + NLTK + Scikit-learn)

A simple FAQ chatbot that matches a user's question to the closest FAQ in
`faqs.csv` using TF-IDF vectorization and cosine similarity, with NLTK used
for text preprocessing (lowercasing, stopword removal, lemmatization).

## Project structure

```
faq_chatbot/
├── app.py               # Flask backend + NLP matching logic
├── faqs.csv              # FAQ dataset (question, answer) — edit/replace this
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Chat UI
└── static/
    └── style.css          # Chat UI styling
```

## How to run in VS Code

1. Open the `faq_chatbot` folder in VS Code.
2. Open a terminal (``Ctrl + ` ``) and create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open your browser at **http://127.0.0.1:5000/**

The first run will auto-download the small NLTK `stopwords` and `wordnet`
data files (needs internet once).

## Customizing the dataset

Edit `faqs.csv` — just keep the two columns `question,answer`. Add as many
rows as you like; the app re-builds the TF-IDF matrix from the CSV every
time it starts, so no other code changes are needed.

## How the matching works

1. Every FAQ question and every user message is preprocessed: lowercased,
   punctuation stripped, stopwords removed, and words lemmatized.
2. All FAQ questions are converted into TF-IDF vectors when the app starts.
3. When a user sends a message, it's converted into a TF-IDF vector using
   the same vocabulary, then compared to every FAQ vector with cosine
   similarity.
4. The FAQ with the highest similarity score is returned as the answer. If
   the best score is below `CONFIDENCE_THRESHOLD` (0.2 in `app.py`), the bot
   replies that it couldn't find a good match instead of guessing.

## Notes for your CodeAlpha submission

- Push this whole folder to a GitHub repo named `CodeAlpha_FAQChatbot`
  (or similar, per the task naming convention).
- Record a short video demoing the chat UI and explaining the
  preprocessing + cosine similarity matching, then post it on LinkedIn
  tagging @CodeAlpha with the repo link.
