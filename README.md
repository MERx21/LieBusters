# 🎭 LieBusters | Real-Time Deception Detection

**LieBusters** is a real-time **multimodal deception detection system** that uses both **audio** and **visual** cues to detect whether someone is lying or telling the truth.

Built with machine learning, deep learning, and a glitchy horror-inspired interface — it’s spooky, smart, and scientifically fun.

---

## 📌 Table of Contents
- [🎯 Features](#-features)
- [🧠 Technologies Used](#-technologies-used)
- [▶️ How to Run](#️-how-to-run)
- [🔐 Model Access](#-model-access)
- [📁 Project Structure](#-project-structure)
- [⚠️ Disclaimer](#️-disclaimer)
- [🪪 License & Author](#-license--author)

---

## 🎯 Features

- 🎥 **Webcam + microphone recording** (via FFmpeg)
- 🎙️ **Audio feature extraction** using OpenSMILE (ComParE_2016)
- 👁️ **Visual embedding** via Vision Transformer (ViT)
- 🤖 **Prediction modules**:
  - Audio: **SVM (scikit-learn)**
  - Visual: **BiGRU (TensorFlow)**
- 🔄 **Fusion logic** to combine modalities
- 🧟 Verdict displayed with a glitch-horror styled UI and spooky sound effects

---

## 🧠 Technologies Used

| Component     | Tools / Libraries |
|---------------|-------------------|
| Backend       | Python 3.10, Flask |
| ML Models     | scikit-learn (SVM), TensorFlow (BiGRU), PyTorch (ViT) |
| Audio Features| OpenSMILE (ComParE_2016) |
| Visual Features | ViT via HuggingFace Transformers |
| Frontend      | HTML, JavaScript, CSS |
| Recording     | FFmpeg |

---

## ▶️ How to Run

1. 📦 **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. 🚀 **Run the Flask app**
   ```bash
   python app.py
   ```

3. 🌐 **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

4. 🎮 **Controls**
   - Press `R` to start recording
   - Press `S` to stop and analyze
   - The system will return a truth or lie verdict onscreen

---

## 🔐 Model Access

The `Models/` directory is not included in the public repository for security reasons.  
Pretrained models are provided in a password-protected archive:

📦 `Models.zip`

To request access, please contact:

📧 **meramtadjine@gmail.com**

Include your name, affiliation, and reason for the request.

---

## 📁 Project Structure

```
LieBusters/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Glitch-style horror UI
├── static/
│   └── audio/          # Sound effects for verdict
├── Models.zip          # 🔐 Pretrained models (protected)
├── requirements.txt    # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
└── recordings/         # Temporary saved recordings (ignored)
```

---

## ⚠️ Disclaimer

This project is part of a controlled academic research study.  
It is not intended for deployment in real-world forensic, legal, or clinical environments.

**Important**: Predictions are limited by dataset quality, recording conditions, and model constraints.  
Use this tool responsibly and only in experimental or educational contexts.

---

## 🪪 License & Author

- **License**: MIT License  
- **Authors**: Meram Tadjine & Ilhem Chaguetmi  
- **Email**: meramtadjine@gmail.com  
- **Year**: 2025  

📜 Redistribution requires appropriate credit and adherence to MIT license terms.

---

👻 *“Are you lying?” — Let the system decide...*
