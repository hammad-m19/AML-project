"""Applied Machine Learning Project - Flask Web Application."""

import os
import socket
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from utils.apriori_mining import run_apriori
from utils.data_preprocessing import analyze_dataset, load_csv, preprocess_for_clustering, preprocess_for_supervised
from utils.dl_models import run_dl_pipeline
from utils.ml_models import run_clustering, run_dbscan, run_ml_pipeline
from utils.transformers_models import answer_question, extract_entities, generate_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "sample_data")

app = Flask(__name__)
app.config["SECRET_KEY"] = "aml-project-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

# In-memory session store for uploaded datasets (per session id)
DATA_STORE = {}


def _get_session_id():
    sid = request.form.get("session_id") or request.args.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
    return sid


def _store_df(session_id, df, filename):
    DATA_STORE[session_id] = {"df": df, "filename": filename}


def _get_df(session_id):
    entry = DATA_STORE.get(session_id)
    if not entry:
        return None
    return entry["df"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ml-algorithms")
def ml_algorithms():
    return render_template("ml_algorithms.html")


@app.route("/deep-learning")
def deep_learning():
    return render_template("deep_learning.html")


@app.route("/generative-ai")
def generative_ai():
    return render_template("generative_ai.html")


@app.route("/voice-qa")
def voice_qa():
    return render_template("voice_qa.html")


@app.route("/text-generation")
def text_generation():
    return render_template("text_generation.html")


@app.route("/ner")
def ner_page():
    return render_template("ner.html")


@app.route("/apriori")
def apriori_page():
    return render_template("apriori.html")


# ── CSV Upload & Analysis ──────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a valid CSV file."}), 400

    try:
        df = load_csv(file)
        session_id = _get_session_id()
        _store_df(session_id, df, secure_filename(file.filename))
        summary = analyze_dataset(df)
        return jsonify({"session_id": session_id, "summary": summary, "filename": file.filename})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/run-ml", methods=["POST"])
def api_run_ml():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    target = data.get("target_column")
    algorithm = data.get("algorithm", "linear_regression")

    df = _get_df(session_id)
    if df is None:
        return jsonify({"error": "Session expired. Please re-upload your CSV."}), 400

    try:
        if algorithm == "kmeans":
            preprocessed = preprocess_for_clustering(df)
            results = run_clustering(preprocessed["X"], n_clusters=int(data.get("n_clusters", 3)))
            results["cluster_info"] = preprocessed["info"]
        elif algorithm == "dbscan":
            preprocessed = preprocess_for_clustering(df)
            results = run_dbscan(
                preprocessed["X"],
                eps=float(data.get("eps", 0.5)),
                min_samples=int(data.get("min_samples", 5)),
            )
            results["cluster_info"] = preprocessed["info"]
        else:
            if not target:
                return jsonify({"error": "Target column is required."}), 400
            preprocessed = preprocess_for_supervised(df, target)
            results = run_ml_pipeline(preprocessed, algorithm)
            results["preprocessing"] = preprocessed["info"]
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/run-dl", methods=["POST"])
def api_run_dl():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    target = data.get("target_column")
    architecture = data.get("architecture", "ann")

    df = _get_df(session_id)
    if df is None:
        return jsonify({"error": "Session expired. Please re-upload your CSV."}), 400
    if not target:
        return jsonify({"error": "Target column is required."}), 400

    try:
        preprocessed = preprocess_for_supervised(df, target)
        results = run_dl_pipeline(
            preprocessed,
            architecture=architecture,
            epochs=int(data.get("epochs", 30)),
            batch_size=int(data.get("batch_size", 32)),
        )
        results["preprocessing"] = preprocessed["info"]
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/run-apriori", methods=["POST"])
def api_run_apriori():
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a valid CSV file."}), 400
        try:
            df = load_csv(file)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        session_id = request.form.get("session_id") or (request.get_json(silent=True) or {}).get("session_id")
        df = _get_df(session_id)
        if df is None:
            return jsonify({"error": "No dataset available. Upload a CSV first."}), 400

    min_support = float(request.form.get("min_support", 0.2))
    min_confidence = float(request.form.get("min_confidence", 0.5))
    min_lift = float(request.form.get("min_lift", 1.0))

    try:
        results = run_apriori(df, min_support, min_confidence, min_lift)
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ── Transformers Endpoints ───────────────────────────────────────────

@app.route("/api/qa", methods=["POST"])
def api_qa():
    data = request.get_json(force=True)
    context = data.get("context", "").strip()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "A question is required."}), 400
    try:
        result = answer_question(question, context)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/text-generate", methods=["POST"])
def api_text_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    try:
        result = generate_text(
            prompt,
            max_length=int(data.get("max_length", 120)),
            num_return_sequences=int(data.get("num_sequences", 1)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ner", methods=["POST"])
def api_ner():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required."}), 400
    try:
        result = extract_entities(text)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/sample/<filename>")
def download_sample(filename):
    from flask import send_from_directory

    allowed = {
        "iris.csv",
        "housing.csv",
        "transactions.csv",
        "qa_context.txt",
    }
    if filename not in allowed:
        return jsonify({"error": "Sample not found."}), 404
    return send_from_directory(SAMPLE_FOLDER, filename, as_attachment=True)


def _find_free_port(preferred=5001):
    """Return preferred port if free, otherwise the next available port."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    if port == 5000:
        port = 5001  # macOS AirPlay Receiver often occupies 5000

    print(f"\n  AML Portal running at: http://localhost:{port}\n")

    try:
        app.run(debug=True, host="0.0.0.0", port=port)
    except OSError:
        port = _find_free_port(port)
        print(f"Port busy — retrying on http://localhost:{port}\n")
        app.run(debug=True, host="0.0.0.0", port=port)
