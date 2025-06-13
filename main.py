import os
import subprocess
import numpy as np
import cv2
import torch
import joblib
import tensorflow as tf
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import pandas as pd

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
record_path = os.path.join(BASE_DIR, "recorded_session.webm")
temp_audio = os.path.join(BASE_DIR, "temp_audio.wav")
temp_csv = os.path.join(BASE_DIR, "temp_audio.csv")

ffmpeg_path = r"C:\Users\meram\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
opensmile_path = r"C:\Users\meram\Downloads\opensmile-3.0-win-x64\bin\SMILExtract.exe"
compare_conf = r"C:\Users\meram\Downloads\opensmile-3.0-win-x64\config\compare16\ComParE_2016.conf"

model_dir = os.path.join(BASE_DIR, "Models")
audio_model = joblib.load(os.path.join(model_dir, "audio_svm_model.pkl"))
audio_scaler = joblib.load(os.path.join(model_dir, "audio_scaler.pkl"))
visual_model = tf.keras.models.load_model(os.path.join(model_dir, "visual_bigru_model_96_TF18_FIXED.h5"))
visual_scaler = joblib.load(os.path.join(model_dir, "visual_scaler_96.pkl"))

# === Record video ===
def record_video(output_path):
    print("\n📹 Recording 5 seconds of webcam + mic...")
    cmd = [
        ffmpeg_path,
        "-y",
        "-f", "dshow",
        "-i", "video=HP HD Camera:audio=Réseau de microphones (Technologie Intel® Smart Sound pour microphones numériques)",
        "-t", "5",
        "-c:v", "libvpx", "-b:v", "1M",
        "-c:a", "libvorbis",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Video saved:", output_path)

# === Extract audio features ===
import opensmile

def extract_audio_features():
    print("🔊 Extracting audio from video...")
    subprocess.run([
        ffmpeg_path, "-y", "-i", record_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio
    ])

    if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
        raise RuntimeError("❌ Audio extraction failed or file is empty.")

    print("🎙️ Running OpenSMILE (Python wrapper)...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )

    features_df = smile.process_file(temp_audio)
    features = features_df.iloc[0].values

    if features.shape[0] != 6373:
        raise ValueError(f"❌ Expected 6373 features, got {features.shape[0]}")

    print("✅ Audio features extracted:", features.shape)
    return audio_scaler.transform([features])

# === Extract visual features ===
def extract_visual_features():
    print("\n🖼️ Extracting visual features from frame...")
    cap = cv2.VideoCapture(record_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("❌ Could not extract frame from video.")

    extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    model.eval()

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    cls_embedding = output.last_hidden_state[:, 0, :].numpy().squeeze()
    print("✅ Visual features extracted:", cls_embedding.shape)
    return visual_scaler.transform([cls_embedding]).reshape(1, 96, 8)

# === Fusion prediction ===
def predict(audio_feat, visual_feat):
    print("\n🤖 Making final prediction...")
    audio_score = audio_model.predict_proba(audio_feat)[0][1]
    visual_score = visual_model.predict(visual_feat)[0][0]
    final_score = (audio_score + visual_score) / 2
    result = "LIAR!" if final_score > 0.5 else "TRUTH!"
    print(f"🧠 Scores — Audio: {audio_score:.2f}, Visual: {visual_score:.2f}, Final: {final_score:.2f}")
    return result

# === Main Runner ===
if __name__ == "__main__":
    print("📼 Starting deception prediction pipeline...")
    record_video(record_path)
    audio_feat = extract_audio_features()
    visual_feat = extract_visual_features()
    result = predict(audio_feat, visual_feat)
    print("\n🎯 Final Prediction:", result)
