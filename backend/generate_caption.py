import os
import io
import base64
import json

import numpy as np
from PIL import Image
import tensorflow as tf

from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import load_model
from pickle import load as pickle_load

# ─── Force CPU only (disable any GPU) ─────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ─── Paths (assumes this file lives in the same folder as tokenizer.p) ────────────
BASE_DIR = os.path.dirname(__file__)
TOKENIZER_PATH       = os.path.join(BASE_DIR, "tokenizer.p")
CAPTION_MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model_9.h5")

# ─── Hyperparameters ──────────────────────────────────────────────────────────────
max_length = 32  # same as used during training

# ─── Load tokenizer and models once at startup ───────────────────────────────────
print("Loading tokenizer...")
tokenizer = pickle_load(open(TOKENIZER_PATH, "rb"))

print("Loading caption model (.h5)...")
caption_model = load_model(CAPTION_MODEL_PATH, compile=False)

print("Loading Xception (feature extractor)...")
xception_model = Xception(include_top=False, pooling="avg")


# ─── Flask setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ─── Utility: Map an integer → word using tokenizer ───────────────────────────────
def word_for_id(integer):
    for w, idx in tokenizer.word_index.items():
        if idx == integer:
            return w
    return None


# ─── Utility: Extract Xception features from a PIL Image ──────────────────────────
def extract_features_from_pil(pil_img):
    """
    1) Resize to 299×299
    2) Preprocess for Xception ([-1..1])
    3) Run through Xception → (1, 2048) vector
    """
    try:
        img = pil_img.resize((299, 299))
        arr = np.array(img).astype("float32")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]  # drop alpha channel if present

        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)  # scale to [-1,1] for Xception
        feats = xception_model.predict(arr, verbose=0)
        return feats  # shape = (1, 2048)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


# ─── Utility: Generate a caption (with "start"/"end") from features ─────────────
def generate_desc(photo_features):
    """
    Given photo_features shape (1,2048), run the LSTM decoder to form:
      "start a dog is running end"
    """
    in_text = "start"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        # Force CPU for LSTM inference
        with tf.device("/CPU:0"):
            yhat = caption_model.predict([photo_features, seq], verbose=0)

        yhat = np.argmax(yhat)
        word = word_for_id(int(yhat))
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break

    return in_text


# ─── Utility: Decode a base64 data URL to a PIL Image ─────────────────────────────
def decode_data_url(data_url):
    """
    Accepts a string like "data:image/jpeg;base64,/9j/4AAQ...". 
    Returns a PIL Image in RGB mode, or None on failure.
    """
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


# ─── Flask route: POST /generate_caption ──────────────────────────────────────────
@app.route("/api/generate_caption", methods=["POST"])
def generate_caption_route():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,/9j/..." }
    Returns JSON: { "caption": "a dog is running" }
    """
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "No image provided"}), 400

    pil_img = decode_data_url(data_url)
    if pil_img is None:
        return jsonify({"error": "Failed to decode base64 image"}), 400

    photo_features = extract_features_from_pil(pil_img)
    if photo_features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    raw_desc = generate_desc(photo_features)
    # strip "start" and "end"
    tokens = raw_desc.split()
    if tokens and tokens[0] == "start":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "end":
        tokens = tokens[:-1]
    final_caption = " ".join(tokens)

    return jsonify({"caption": final_caption})


# ─── Run locally: python app.py ────────────────────────────────────────────────────
if __name__ == "__main__":
    # By default Flask listens on 127.0.0.1:5000
    # Set debug=True if you want automatic reload on code changes
    app.run(host="0.0.0.0", port=5000, debug=True)
