# Applied Machine Learning Project

Flask-based web portal for Machine Learning, Deep Learning, Generative AI (Transformers), and Association Rule Mining.

## Features

### Part (a) — Three Main Sections
1. **Machine Learning Algorithms** — Upload CSV, preprocess data, run Logistic Regression, Random Forest, SVM, Linear Regression, K-Means
2. **Deep Learning Algorithms** — Feed-forward neural network with training visualization
3. **Generative AI (Transformers)** — Links to transformer-based NLP modules

### Part (b) — Required Functional Models
1. **Voice Question Answering** — Speech-to-text input, DistilBERT QA, text-to-speech output
2. **Text Generation** — DistilGPT-2 text-in/text-out generation
3. **Named Entity Recognition** — BERT-based NER with interactive highlighting
4. **Apriori Mining** — Association rules from uploaded transaction CSV

## Setup

```bash
cd "Project"
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5001** in your browser.

> On macOS, port 5000 is often used by AirPlay Receiver. This app defaults to **5001**. To use a custom port: `PORT=8080 python app.py`

> **Note:** First run downloads Hugging Face transformer models (~500MB). Voice QA requires Chrome or Edge for Web Speech API.

## Sample Datasets

| File | Use Case |
|------|----------|
| `sample_data/iris.csv` | ML classification (target: `species`) |
| `sample_data/housing.csv` | ML/DL regression (target: `MEDV`) |
| `sample_data/transactions.csv` | Apriori association rules |

## Project Structure

```
Project/
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

- **Backend:** Flask, scikit-learn, TensorFlow/Keras, Hugging Face Transformers, mlxtend
- **Frontend:** HTML/CSS/JS, Plotly.js, Web Speech API
- **Visualization:** Interactive Plotly charts (confusion matrix, correlation heatmap, training curves, NER distribution)
