# ─── pa_backend/app.py ───────────────────────────────────────────────────────────

import os
import io
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from PIL import Image
import onnxruntime as ort
from pickle import load as pickle_load
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Force CPU only ──────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─── Flask setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/generate_caption": {"origins": "*"}})

# ─── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.p")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_9.onnx")

# ─── Load tokenizer at cold‐start ────────────────────────────────────────────────
tokenizer = pickle_load(open(TOKENIZER_PATH, "rb"))
max_length = 32

# ─── Load ONNX session at cold‐start ──────────────────────────────────────────────
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
input_names = [inp.name for inp in session.get_inputs()]
output_name = session.get_outputs()[0].name

# ─── Decode base64 data URL → PIL Image ──────────────────────────────────────────
def decode_data_url(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

# ─── Stub for feature extraction ─────────────────────────────────────────────────
# You must replace this stub with a real ONNX encoder (e.g. xception.onnx).
def extract_features_from_pil(pil_img):
    # Resize to 299×299, etc., then run your ONNX‐ified encoder.
    # For now, we return zeros so the app runs:
    return np.zeros((1, 2048), dtype=np.float32)

# ─── Map integer → word via tokenizer ───────────────────────────────────────────
def word_for_id(integer):
    for w, idx in tokenizer.word_index.items():
        if idx == integer:
            return w
    return None

# ─── Generate a caption string via ONNX LSTM decoder ─────────────────────────────
def generate_desc_onnx(photo_features):
    in_text = "start"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length).astype(np.int64)

        ort_inputs = {
            input_names[0]: photo_features.astype(np.float32),
            input_names[1]: seq
        }
        ort_outs = session.run([output_name], ort_inputs)
        yhat = np.argmax(ort_outs[0])
        word = word_for_id(int(yhat))
        if not word or word == "end":
            break
        in_text += " " + word

    return in_text

# ─── Flask route: /generate_caption ───────────────────────────────────────────────
@app.route("/generate_caption", methods=["POST"])
def generate_caption_route():
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "No image provided"}), 400

    pil_img = decode_data_url(data_url)
    if pil_img is None:
        return jsonify({"error": "Invalid base64 image"}), 400

    photo_features = extract_features_from_pil(pil_img)
    if photo_features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    raw_desc = generate_desc_onnx(photo_features)
    tokens = raw_desc.split()
    if tokens and tokens[0] == "start":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "end":
        tokens = tokens[:-1]
    final_caption = " ".join(tokens)

    return jsonify({"caption": final_caption})

# ─── (Optional) Run via `python app.py` for local testing ─────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
