# 🛡️ STCA-Net Deepfake Detection Project

A novel, lightweight deepfake detection system utilizing a **Spatio-Temporal Cross-Attention (STCA)** architecture. This project efficiently combines the spatial feature extraction of MobileNetV3 with the global context encoding of a Vision Transformer (ViT) to detect pixel-level and context-level artifacts in both images and videos.

## ✨ Features

- **Advanced Image & Video Analysis**: Uses OpenCV/Haar Cascades to reliably detect and crop faces before feeding them into the neural network, drastically improving accuracy by ignoring irrelevant backgrounds.
- **Frequency-Domain AI Detection**: Identifies AI-generated images (GANs, diffusion) by combining cross-attention neural network scores with Discrete Cosine Transform (DCT) frequency spectrum analysis.
- **Non-Photographic Recognition**: Prevents false predictions on cartoons, anime, and illustrations by evaluating color variance and edge densities.
- **Optimized for CPU**: Designed for researchers and developers with limited hardware. The STCA-Net architecture boasts under 10MB of trainable parameters.
- **Modern User Interface**: A responsive, Flask-powered web dashboard that visualizes prediction confidence, processing time, and frequency analysis outcomes.

## 📁 Repository Structure

```
STCA-Net-Project -og/
├── app.py                     # Main Flask web server
├── train_stca_net.py          # Script to train/finetune the model
├── requirements.txt           # Python dependencies (CPU/CUDA)
├── models/
│   └── stca_net.py            # The PyTorch model architecture
├── utils/
│   ├── prediction.py          # Image & video analysis pipeline (NN + Frequency)
│   └── video_processing.py    # OpenCV face extraction & frame sampling
├── static/
│   └── style.css              # Modern UI/UX styles
└── templates/                 # HTML templates (index, detect, image)
```

> **Note**: This repository contains only the code. Large datasets, heavy virtual environments, and the compiled model weights (`.pt` files) are intentionally excluded via `.gitignore` to keep the repository lightweight.

## 🚀 Getting Started

### 1. Installation

Requires Python 3.10+ (Tested up to 3.13).

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "STCA-Net-Project -og"

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Note: If you run into `dlib` installation issues on Windows, `app.py` has been rewritten to prioritize OpenCV face detection instead).*

### 2. Training the Model

Before running the web application, you should train the model on a deepfake dataset (e.g., FaceForensics++, Celeb-DF, etc.). Place your dataset in a `dataset/` folder with `real` and `fake` subdirectories containing images.

```bash
# Train the model (automatically saves to model/stca_net_weights.pt)
python train_stca_net.py --dataset path/to/dataset --epochs 15 --samples 10000 --unfreeze-layers 3
```

### 3. Running the Web App

Once the model is trained:

```bash
python app.py
```

Open your browser to `http://127.0.0.1:5000` to access the Deepfake Scanner Dashboard.

## 🧠 Architectural Overview

Our STCA-Net utilizes a hybrid approach:
1. **Local Extractor**: A pre-trained MobileNetV3-Small backbone extracts spatial features (edges, textures, pixel blending artifacts).
2. **Global Encoder**: A 2-layer Vision Transformer (ViT) processes the spatial map as a sequence to understand global context and inconsistencies across the face.
3. **Cross-Attention Fusion**: Attention mechanisms merge the global anomalies with specific spatial regions for final classification.
4. **Frequency Fallback**: Pixel-based neural networks struggle with the smooth spectral decay of Latent Diffusion models. STCA-Net combines NN scores with a traditional Discrete Cosine Transform evaluating high-frequency energy ratios.

## 📄 License
MIT License. Feel free to fork and modify for research or educational purposes.
