# Customer Sentiment Analysis App

Predicts whether a product review is **Positive**, **Negative**, or **Neutral** using a machine learning model trained on Amazon product reviews. Built as part of the AnalystLab Africa Artificial Intelligence Internship (Weeks 5, 7 & 8 Capstone).

🔗 **Live app:** https://sentiment-analysis-owonubi-michael.streamlit.app
📓 **Notebook:** `Week5_NLP_Sentiment_Analysis_Amazon_Reviews.ipynb`

---

## Project Description

Businesses receive large volumes of customer reviews and can't manually read every one to gauge sentiment. This project automates that process: it takes raw review text, cleans and vectorizes it, and classifies the sentiment behind it — allowing negative feedback to be flagged and trends to be tracked at scale.

## Problem Being Solved

Manually monitoring customer sentiment across thousands of reviews is slow and inconsistent. This app gives a fast, automated way to classify sentiment so teams can prioritize responses and spot patterns in customer feedback.

## Dataset

- **Source:** Real Amazon product reviews (electronics category)
- **Labels:** Derived from star ratings — 1–2★ = Negative, 3★ = Neutral, 4–5★ = Positive
- **Sample size used:** 8,000 reviews

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| NLP preprocessing | NLTK |
| Vectorization | TF-IDF (scikit-learn) |
| Model | Logistic Regression (class_weight='balanced') |
| Visualization | Matplotlib, Seaborn |
| App framework | Streamlit |
| Model persistence | Pickle |

## How the AI System Works

1. **Text preprocessing** — input text is lowercased, stripped of punctuation and special characters, and stopwords are removed.
2. **Vectorization** — cleaned text is converted into numerical features using TF-IDF.
3. **Prediction** — the trained classifier predicts sentiment (Positive / Negative / Neutral) from the vectorized input.
4. **Output** — the app displays the predicted sentiment.

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 87.5% |
| Precision (weighted) | 91.61% |
| Recall (weighted) | 87.5% |
| F1-Score (weighted) | 89.34% |

**Class-level breakdown (test set):**

| Class | Correctly Identified |
|---|---|
| Negative | 16 of 37 |
| Neutral | 20 of 65 |
| Positive | 1,364 of 1,498 |

## Repository Structure

```
├── Week5_NLP_Sentiment_Analysis_Amazon_Reviews.ipynb   # full pipeline: cleaning, EDA, TF-IDF, model training, evaluation
├── model.pkl                       # saved trained model
├── vectorizer.pkl                  # saved TF-IDF vectorizer
├── app.py                          # Streamlit application
├── requirements.txt                # project dependencies
└── README.md
```

## Running the Project Locally

1. Clone the repository:
   ```bash
   git clone [your repo URL]
   cd [repo name]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Using the Deployed App

1. Visit the live app link above.
2. Type a product review into the text box.
3. Click **Predict Sentiment** to see the result.

## Challenges Faced

- The first trained model achieved high accuracy (93.75%) but turned out to be almost entirely predicting "Positive" due to severe class imbalance in the training data (1,498 Positive vs. only 37 Negative and 65 Neutral reviews).
- Fixed this by retraining with `class_weight='balanced'`, which meaningfully improved the model's ability to detect Negative and Neutral sentiment, at the cost of a slightly lower overall accuracy (87.5%).
- Several dataset download links were unstable during development (dead GitHub links, SSL certificate errors) and required troubleshooting alternate sources.

## Key Learnings

- Accuracy alone can be misleading on imbalanced datasets — the confusion matrix was what revealed the real problem.
- How TF-IDF represents text numerically for classical ML models.
- How `class_weight='balanced'` in scikit-learn changes what a model optimizes for, and why a lower accuracy score can reflect a genuinely more useful model.

## Future Improvements

- Train on the full dataset rather than an 8,000-review sample for stronger generalization.
- Experiment with word embeddings or a transformer-based model.
- Further improve minority-class detection with oversampling techniques.
- Deploy as a REST API for real-time integration into other systems.

---

*Built by Owonubi Michael Favour as part of the AnalystLab Africa Artificial Intelligence Internship Program.*
