# LieBusters Deception Detection System
# Author: Meram Tadjine & Ilhem Chaguetmi
# Email: meramtadjine@gmail.com
# License: MIT (see LICENSE file)
# === Standard Library Imports ===
import os
import subprocess
import shutil
import datetime
import warnings
import logging

# === Suppress Warnings and TensorFlow Logs ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('absl').setLevel(logging.ERROR)

# === Third-Party Imports ===
import numpy as np
import cv2
import torch
import joblib
import pandas as pd
import tensorflow as tf
import opensmile

from PIL import Image
from flask import Flask, render_template, jsonify, request
from transformers import ViTFeatureExtractor, ViTModel
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# === Flask App Initialization ===
app = Flask(__name__)

# === Path Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
record_path = os.path.join(BASE_DIR, "recorded_session.mp4")
temp_audio = os.path.join(BASE_DIR, "temp_audio.wav")
ffmpeg_path = r"C:\Users\meram\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# === Load Pretrained Models ===
model_dir = os.path.join(BASE_DIR, "Models")
audio_model = joblib.load(os.path.join(model_dir, "audio_svm_model.pkl"))
audio_scaler = joblib.load(os.path.join(model_dir, "audio_scaler.pkl"))
visual_model = tf.keras.models.load_model(os.path.join(model_dir, "visual_bigru_model_96_TF18_FIXED.h5"))
visual_scaler = joblib.load(os.path.join(model_dir, "visual_scaler_96.pkl"))

# === OpenSMILE Audio Feature Extractor ===
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

# === Route: Home Page ===
@app.route("/")
def index():
    return render_template("index.html")

# === Route: Record from Webcam (Optional) ===
@app.route("/record", methods=["POST"])
def record_video():
    print("📹 Recording 10 seconds of webcam + mic...")

    # Remove old recording
    if os.path.exists(record_path):
        os.remove(record_path)

    # Run ffmpeg to capture 10 seconds of video+audio
    cmd = [
        ffmpeg_path, "-y",
        "-f", "dshow",
        "-video_size", "1280x720",
        "-i", 'video=GENERAL - UVC:audio=Réseau de microphones (Technologie Intel® Smart Sound pour microphones numériques)',
        "-t", "10",
        "-c:v", "libvpx", "-b:v", "1M",
        "-c:a", "libvorbis",
        record_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Video saved:", record_path)
    return jsonify({"status": "recorded"})

# === Route: Predict From Uploaded Video ===
@app.route("/predict", methods=["POST"])
def predict_label():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    # Save uploaded video to 'recordings' folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    recordings_dir = os.path.join(BASE_DIR, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    uploaded_path = os.path.join(recordings_dir, f"video_{timestamp}.webm")
    request.files["video"].save(uploaded_path)
    shutil.copy(uploaded_path, record_path)
    print("📁 New video saved to:", uploaded_path)

    # === Step 1: Extract Audio Features ===
    print("🔊 Extracting audio from video...")
    subprocess.run([
        ffmpeg_path, "-y", "-i", record_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
        return jsonify({"error": "Audio extraction failed"}), 500

    print("🎙️ Running OpenSMILE...")
    features_df = smile.process_file(temp_audio)
    features = features_df.iloc[0].values

    if features.shape[0] != 6373:
        return jsonify({"error": "Wrong audio feature length", "shape": str(features.shape)}), 500

    audio_input = audio_scaler.transform([features])

    # === Step 2: Extract Visual Features ===
    print("🖼️ Extracting visual features...")
    cap = cv2.VideoCapture(record_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Frame capture failed"}), 500

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_model.eval()

    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = vit_model(**inputs)

    visual_input = visual_scaler.transform(
        [output.last_hidden_state[:, 0, :].numpy().squeeze()]
    ).reshape(1, 96, 8)

    # === Step 3: Predict Deception ===
    audio_score = audio_model.predict_proba(audio_input)[0][1]
    visual_score = visual_model.predict(visual_input)[0][0]
    final_score = (audio_score + visual_score) / 2
    label = 1 if final_score > 0.5 else 0

    print(f"🧠 Scores — Audio: {audio_score:.2f}, Visual: {visual_score:.2f}, Final: {final_score:.2f}")
    print("🎯 Final Prediction:", "LIAR" if label else "TRUTH")

    return jsonify({
        "label": label,
        "audio_score": float(audio_score),
        "visual_score": float(visual_score),
        "final_score": float(final_score)
    })

# === Start the Server ===
if __name__ == "__main__":
    app.run(debug=True)
