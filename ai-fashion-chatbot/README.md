# 👗 AI Fashion Chatbot — Multimodal Visual Search & Recommendation

An end-to-end AI-powered fashion assistant built on **real scraped data from Jumia Egypt**. Upload any clothing photo and the system generates a natural-language caption and retrieves the most visually similar products from the catalog using deep learning embeddings.

---

## 🗺️ Full Pipeline Overview

```
Jumia Egypt
    ↓
① Web Scraping (Selenium)          jumia_scraper.ipynb
    ↓ products.csv (~5,955 products)
② Text Preprocessing               preprocessing.ipynb
    ↓ cleaned_data_final.csv
③ Image Downloading & Validation   image_preprocessing.ipynb
    ↓ images_fresh/{product_id}.jpg
④ Model Training (4 architectures) Model1.ipynb / Models.ipynb / model2.ipynb / model_3.ipynb
    ↓ saved_models/
⑤ Visual Retrieval System          retrieval_system.ipynb
    ↓ embeddings_cache/
⑥ Chatbot Interface                chatbot.ipynb
    ↓
User uploads image → Caption + Top-5 similar products
```

---

## ✨ Features

- 🕷️ **Custom Web Scraper** — Scrapes 35 fashion categories from Jumia Egypt (Men, Women, Kids)
- 🧹 **Full Data Pipeline** — Price parsing, deduplication, category normalization, color & brand extraction
- 📥 **Parallel Image Downloader** — Multi-threaded with retry logic and broken-image detection
- 🧠 **4 Caption Models** — Trained and compared side by side
- 🔍 **Visual Similarity Search** — Cosine similarity on EfficientNetB0 embeddings (FAISS optional)
- 💬 **Interactive Chatbot** — Jupyter widget interface with image upload, caption, and product results

---

## 🗂️ Dataset

**Source:** Jumia Egypt — scraped with Selenium

| Property | Value |
|----------|-------|
| Total Products | ~5,955 |
| Categories | 35 scraped → 26 normalized |
| Genders | Men, Women, Kids |
| Fields | title, price (EGP), category, gender, img\_url, product\_url |

**Price Buckets:**

| Bucket | Range (EGP) |
|--------|-------------|
| Budget | < 300 |
| Mid-range | 300 – 799 |
| Premium | 800 – 1,999 |
| Luxury | 2,000+ |

**Scraped Categories include:**
Women's dresses, jeans, shoes, bags, hoodies, skirts, coats, watches, jewelry, belts, sandals, boots, sneakers, sunglasses, wallets — and equivalent men's and kids' categories.

---

## 🤖 Model Architectures

All 4 models follow the same pattern: **CNN backbone → feature vector → sequence decoder → caption**

| Model | CNN Backbone | Decoder | Attention |
|-------|-------------|---------|-----------|
| 1 | ResNet50V2 (2048-d) | LSTM | Bahdanau |
| 2 | ResNet50 (2048-d) | GRU | Bahdanau |
| 3 | EfficientNetB0 (1280-d) | LSTM | Bahdanau |
| 4 ⭐ | EfficientNetB0 (1280-d) | Transformer Decoder | Multi-Head (4 heads) |

**Model 4 is used in production** (retrieval + chatbot) for best performance.

### Model 4 Architecture Detail

```
Input Image (224×224×3)
        ↓
EfficientNetB0 — frozen (ImageNet weights)
        ↓
GlobalAveragePooling2D → 1280-d feature vector
        ↓
Dense projection → embed_dim (512)
        ↓
Transformer Decoder Block
    ├── Self-Attention (4 heads, causal mask)
    ├── Cross-Attention (4 heads, image features)
    └── Feed-Forward (GELU, 512 units)
        ↓
Dense (vocab_size=4000, softmax)
        ↓
Generated Caption
```

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Image Size | 224 × 224 |
| Vocab Size | 4,000 tokens |
| Max Caption Length | 32 tokens |
| Embed Dim | 512 |
| Transformer Heads | 4 |
| FF Dim | 512 |
| Dropout | 0.3 |
| Batch Size | 16 |
| Max Epochs | 15 (EarlyStopping) |
| Optimizer | Adam (lr=1e-3) |
| Data Split | 80% train / 10% val / 10% test |

### Evaluation Metrics
- **BLEU Score** (NLTK) — n-gram caption quality
- **ROUGE Score** — recall-oriented caption overlap

---

## 🔍 Retrieval System

```
User Image
    ↓
EfficientNetB0 → 1280-d embedding
    ↓
Cosine Similarity vs all product embeddings
    ↓
Top-5 most similar products
    ↓
Caption + metadata + product URL
```

- Product embeddings pre-computed and cached as `.npy` files
- Optional **FAISS** index for fast approximate nearest-neighbor search
- Falls back to sklearn cosine similarity if FAISS not installed

---

## 💬 Chatbot Interface

Built with **ipywidgets** inside Jupyter:

1. Upload a clothing image (local file or URL)
2. EfficientNetB0 + Transformer generates a natural-language caption
3. System retrieves **Top-5 visually similar products**
4. Results shown with: similarity score, title, price, category, gender, product link

**Example caption output:**
```
<start> women black hoodie sweatshirt tops mid-range egp 450 <end>
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Scraping | Selenium, webdriver-manager, requests |
| Data | Pandas, NumPy, ftfy, unidecode |
| Deep Learning | TensorFlow / Keras |
| CNN Backbones | ResNet50V2, ResNet50, EfficientNetB0 |
| Sequence Models | LSTM, GRU, Transformer Decoder |
| Attention | Bahdanau Attention, Multi-Head Attention |
| Retrieval | Cosine Similarity (sklearn), FAISS (optional) |
| Metrics | BLEU (NLTK), ROUGE |
| Image | Pillow, scikit-image |
| UI | ipywidgets, Matplotlib |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
ai-fashion-chatbot/
│
├── jumia_scraper.ipynb          ← Selenium scraper for Jumia Egypt
├── preprocessing.ipynb          ← Text cleaning, price parsing, feature engineering
├── image_preprocessing.ipynb    ← Parallel image downloader & validator
│
├── Model1.ipynb                 ← All 4 models: training & comparison
├── Models.ipynb                 ← Model variations
├── model2.ipynb                 ← ResNet50 + GRU variant
├── model_3.ipynb                ← EfficientNetB0 + LSTM variant
│
├── retrieval_system.ipynb       ← Visual similarity search engine
├── chatbot.ipynb                ← Interactive chatbot interface
│
├── products.csv                 ← Raw scraped data (~5,955 products)
├── cleaned_data_final.csv       ← Cleaned & feature-engineered dataset
│
├── saved_models/
│   ├── model4_efficientnetb0_transformer_best.keras
│   └── tokenizer.json
│
├── embeddings_cache/
│   ├── product_embeddings.npy
│   └── product_ids.npy
│
└── data/
    └── images_fresh/            ← Downloaded product images
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow keras numpy pandas pillow scikit-learn matplotlib tqdm
pip install selenium webdriver-manager requests ftfy unidecode
pip install nltk rouge-score ipywidgets
pip install faiss-cpu  # optional — for fast retrieval
```

### Run Order

```
1. jumia_scraper.ipynb          → generates products.csv
2. preprocessing.ipynb          → generates cleaned_data_final.csv
3. image_preprocessing.ipynb    → downloads images to data/images_fresh/
4. Model1.ipynb                 → trains all 4 models, saves to saved_models/
5. retrieval_system.ipynb       → pre-computes & caches embeddings
6. chatbot.ipynb                → interactive chatbot interface
```

> ⚠️ Update `IMG_DIR`, `CSV_PATH`, and `MODELS_DIR` paths in each notebook to match your local environment.

---

## 📦 Code Files

<!-- Add your project files here -->
