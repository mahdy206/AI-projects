#  Data Mining Project — Movie Watch History Analysis

A comprehensive data mining project applied on a **Netflix-inspired movie watch history dataset** (7,501 user sessions). The project covers Association Rule Mining, Graph-based Movie Ranking, and Sentiment Analysis using state-of-the-art NLP models.

---

##  Project Sections

| # | Section | Technique |
|---|---------|-----------|
| 1 | Association Rule Mining | Apriori Algorithm |
| 2 | Frequent Pattern Mining | FP-Growth (Closed & Maximal Itemsets) |
| 3 | Movie Ranking | PageRank on Co-Watch Graph |
| 4 | Sentiment Analysis | BERT (fine-tuned on IMDB reviews) |

---

##  Dataset

### Movies Dataset (`movies_dataset.csv`)
| Property | Value |
|----------|-------|
| Sessions | 7,501 user watch sessions |
| Movies per session | Up to 10 movies |
| Source | Netflix-inspired synthetic dataset |
| Genres covered | Action, Sci-Fi, Drama, Crime, Animation, Romance, Fantasy, Thriller |

### IMDB Dataset (`IMDB Dataset.csv`)
| Property | Value |
|----------|-------|
| Task | Sentiment Classification |
| Labels | Positive / Negative |
| Train size | 1,000 reviews (500 per class) |

---

##  Section 1 — Apriori Algorithm

Discovers **association rules** between movies watched together in the same session.

**Parameters used:**
| Parameter | Value |
|-----------|-------|
| min_support | 0.003 |
| min_confidence | 0.2 |
| min_lift | 3 |
| min_length | 2 |

**Output:** Rules sorted by Lift — e.g., *"Users who watched Inception also watched Interstellar"*

---

##  Section 2 — FP-Growth

A faster alternative to Apriori that mines frequent itemsets using a tree structure.

- **Frequent Itemsets** — All itemsets above `min_support = 0.01`
- **Closed Itemsets** — No superset with the same support exists
- **Maximal Itemsets** — No frequent superset exists at all
- **Association Rules** generated with `min_lift = 1`

---

##  Section 3 — PageRank on Co-Watch Graph

Builds a **directed weighted graph** where:
- **Nodes** = Movies
- **Edges** = Co-watch frequency between two movies in the same session

Applies **PageRank** to rank movies by their influence in the viewing network — similar to how Google ranks pages.

---

##  Section 4 — BERT Sentiment Analysis

Fine-tunes **BERT (bert-base-uncased)** on IMDB movie reviews for binary sentiment classification.

**Pipeline:**
```
Raw Review Text
    ↓
HTML Cleaning + Whitespace Normalization
    ↓
BERT Tokenizer (max_length=128, padding, truncation)
    ↓
BertForSequenceClassification (2 labels)
    ↓
Fine-tuning (2 epochs, batch_size=8)
    ↓
Positive  / Negative  + Confidence %
```

**Example output:**
```
Review   : This movie was absolutely amazing, best film I have seen in years!...
Result   : Positive 
Confidence: 94.3%
```

---

##  Visualizations

| Chart | Description |
|-------|-------------|
| Genre Distribution | Bar + Pie chart of genre watch counts |
| Top 20 Movies | Horizontal bar chart of most-watched movies |
| Rules Scatter Plot | Support vs Confidence colored by Lift |

---

##  Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3 |
| Data | Pandas, NumPy |
| Mining | apyori, mlxtend (FP-Growth) |
| Graph | NetworkX |
| NLP | HuggingFace Transformers (BERT), PyTorch |
| Visualization | Matplotlib |
| Notebook | Jupyter |

---

##  Getting Started

```bash
pip install numpy pandas matplotlib apyori mlxtend networkx
pip install transformers torch scikit-learn
```

Then open the notebook:
```bash
jupyter notebook DM.ipynb
```

Make sure `movies_dataset.csv` and `IMDB Dataset.csv` are in the same directory.

---

## 📦 Code Files

<!-- Add your project files here (DM.ipynb, movies_dataset.csv, IMDB Dataset.csv) -->
