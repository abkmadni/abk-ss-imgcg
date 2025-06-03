# ─── backend/generate_caption.py ───────────────────────────────────────────────
import os
import io
import sys
import base64
import json
import tempfile
import numpy as np
from PIL import Image
import onnxruntime as ort
import urllib.request
from pickle import load as pickle_load
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paths & URLs
BASE_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.p")

# Public URL where you host best_model_9.onnx
MODEL_URL = "https://my-bucket.s3.amazonaws.com/best_model_9.onnx"

# Local path in the serverless container’s /tmp
LOCAL_ONNX_PATH = os.path.join(tempfile.gettempdir(), "best_model_9.onnx")

# Load tokenizer once at cold‐start
tokenizer = pickle_load(open(TOKENIZER_PATH, "rb"))
max_length = 32

# A helper to download the ONNX if not already in /tmp
def ensure_model_downloaded():
    if os.path.exists(LOCAL_ONNX_PATH):
        return

    # Download into /tmp
    try:
        with urllib.request.urlopen(MODEL_URL) as resp:
            data = resp.read()
            with open(LOCAL_ONNX_PATH, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"[ERROR] Could not download ONNX model: {e}", file=sys.stderr)
        raise

# Initialize ONNX Runtime session lazily
_session = None
_input_names = None
_output_name = None

def get_onnx_session():
    global _session, _input_names, _output_name
    if _session is None:
        # Ensure file is present
        ensure_model_downloaded()
        # Now load
        _session = ort.InferenceSession(LOCAL_ONNX_PATH, providers=["CPUExecutionProvider"])
        _input_names = [inp.name for inp in _session.get_inputs()]
        _output_name = _session.get_outputs()[0].name
    return _session, _input_names, _output_name

# Decode base64 data URL → PIL Image
def decode_data_url(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

# Stub for feature extraction — **YOU MUST replace** this with your own ONNX or small CNN.
def extract_features_from_pil(pil_img):
    # Example: return zeros so code runs; replace with your own ONNX-based encoder
    return np.zeros((1, 2048), dtype=np.float32)

# Map integer → word via tokenizer
def word_for_id(integer):
    for w, idx in tokenizer.word_index.items():
        if idx == integer:
            return w
    return None

# Run the ONNX LSTM decoder
def generate_desc_onnx(photo_features):
    session, input_names, output_name = get_onnx_session()

    in_text = "start"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        seq = seq.astype(np.int64)

        ort_inputs = {
            input_names[0]: photo_features.astype(np.float32),
            input_names[1]: seq
        }
        ort_outs = session.run([output_name], ort_inputs)
        yhat = np.argmax(ort_outs[0])
        word = word_for_id(int(yhat))
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break

    return in_text

# Vercel handler
def handler(request):
    if request.method != "POST":
        return {"statusCode": 405, "body": json.dumps({"error": "Method Not Allowed"})}

    ct = request.headers.get("content-type", "")
    if "application/json" not in ct.lower():
        return {"statusCode": 400, "body": json.dumps({"error": "Content-Type must be application/json"})}

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return {"statusCode": 400, "body": json.dumps({"error": "No image provided"})}

    pil_img = decode_data_url(data_url)
    if pil_img is None:
        return {"statusCode": 400, "body": json.dumps({"error": "Failed to decode base64 image"})}

    photo_features = extract_features_from_pil(pil_img)
    if photo_features is None:
        return {"statusCode": 500, "body": json.dumps({"error": "Feature extraction failed"})}

    raw_desc = generate_desc_onnx(photo_features)
    tokens = raw_desc.split()
    if tokens and tokens[0] == "start":
        tokens = tokens[1:]
    if tokens and tokens[-1] == "end":
        tokens = tokens[:-1]
    final_caption = " ".join(tokens)

    return {
        "statusCode": 200,
        "body": json.dumps({"caption": final_caption})
    }
