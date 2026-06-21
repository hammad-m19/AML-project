# Applied Machine Learning Project

Flask-based web portal for Machine Learning, Deep Learning, Generative AI (Transformers), and Association Rule Mining.

## Features

### Part (a) — Three Main Sections
1. **Machine Learning Algorithms** — Upload CSV, preprocess data, run Linear Regression, K-Means Clustering, and DBSCAN Clustering.
2. **Deep Learning Algorithms** — Train ANN, LSTM, and CNN architectures on tabular CSV data with live training curves visualization.
3. **Generative AI (Transformers)** — Interactive dashboard highlighting the NLP models and capabilities of the portal.

### Part (b) — Required Functional Models
1. **Voice Question Answering** — Speech-to-text input, QA processing with a local **Flan-T5** model (with fallback to Mistral-7B-Instruct API), and spoken text-to-speech output.
2. **Text Generation** — Creative text generation with **DistilGPT-2** transformer model.
3. **Named Entity Recognition** **(NER)** — Extract Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC) entities with **dslim/bert-base-NER**.
4. **Apriori Mining** — Generate association rules (antecedents, consequents, support, confidence, lift) from transaction CSV data.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5001** in your browser.

> **Note:** First run downloads Hugging Face transformer models (~500MB). Voice QA requires Chrome or Edge for Web Speech API.

## Sample Datasets

| File | Use Case |
|------|----------|
| `sample_data/iris.csv` | ML classification/clustering (target: `species`) |
| `sample_data/housing.csv` | ML/DL regression (target: `MEDV`) |
| `sample_data/transactions.csv` | Apriori association rules |

## Project Structure

```
AML-project/
├── app.py                  # Flask application
├── requirements.txt
├── utils/
│   ├── data_preprocessing.py
│   ├── ml_models.py
│   ├── dl_models.py
│   ├── transformers_models.py
│   └── apriori_mining.py
├── templates/              # HTML pages
├── static/                 # CSS & JavaScript
└── sample_data/            # Sample CSV files
```

## Tech Stack

- **Backend:** Flask, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, mlxtend
- **Frontend:** HTML/CSS/JS, Plotly.js, Web Speech API
- **Visualization:** Interactive Plotly charts (confusion matrix, correlation heatmap, training curves, NER distribution)

