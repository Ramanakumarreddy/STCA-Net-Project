# 🛡️ STCA-Net — Deepfake Detection System

> **Spatio-Temporal Cross-Attention Network** for detecting deepfake images and videos using a hybrid neural + frequency-domain approach.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

STCA-Net is a lightweight, novel deepfake detection architecture that hybridizes:

1. **CNN spatial feature extraction** (MobileNetV3-Small backbone) for capturing pixel-level artifacts.
2. **Vision Transformer global context encoding** for detecting structural inconsistencies across a face.
3. **Frequency-domain analysis** (FFT + DCT) to identify the spectral fingerprints of AI-generated content.
4. **Metadata & signature checks** to catch obvious AI-generation watermarks in filenames and EXIF data.

The result is a system capable of analyzing both **still images** and **video clips** through a clean Flask web dashboard, while being efficient enough to run on CPU-only hardware.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Face-Focused Analysis** | OpenCV Haar Cascade detects and crops faces before neural processing, filtering out irrelevant backgrounds. |
| **Frequency-Domain AI Detection** | DCT band-energy analysis identifies GAN/diffusion model spectral signatures (low high-frequency energy, steep spectral decay). |
| **Non-Photographic Guard** | Saturation, edge density, and texture variance heuristics prevent false predictions on cartoons and anime. |
| **AI Signature Detection** | Checks filenames and EXIF metadata for known AI tool watermarks (Gemini, DALL-E, Midjourney, Stable Diffusion). |
| **Temporal Video Analysis** | Processes videos as frame sequences through the temporal encoder for coherent multi-frame decisions. |
| **CPU Optimized** | Under 10 MB of trainable parameters; designed to run without a GPU. |
| **Web Dashboard** | Flask-powered UI with confidence gauges, per-frame score timelines, and frequency analysis readouts. |

---

## 📁 Repository Structure

```
STCA-Net-Project/
│
├── app.py                        # Main Flask web application entry point
├── server.py                     # Lightweight server wrapper (used by Docker/Render)
│
├── models/
│   ├── stca_net.py               # PyTorch model architecture (STCANet class)
│   └── stca_net_weights.pt       # Trained model weights (not committed to git)
│
├── utils/
│   ├── prediction.py             # Image & video inference pipeline
│   └── video_processing.py       # Smart frame extraction with face detection
│
├── templates/                    # Jinja2 HTML templates
│   ├── index.html                # Landing page
│   ├── detect.html               # Video analysis page
│   └── image.html                # Image analysis page
│
├── static/
│   └── style.css                 # Application CSS styles
│
├── train_stca_net.py             # Full training script (standard mode)
├── train_in_chunks.py            # Chunk-based progressive trainer with replay buffer
├── process_raw_videos.py         # Dataset builder: extracts frames from raw video files
├── eval_comprehensive.py         # End-to-end evaluation across the full Celeb-DF-v2 dataset
├── validate_model.py             # Quick validation on chunk_data / benchmark_data directories
├── validate_weights.py           # Sanity check: verifies the saved weights are loadable
│
├── Celeb-DF-v2/                  # Dataset directory (not committed to git)
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   └── YouTube-real/
│
├── Uploaded_Files/               # Runtime uploads from the web app (auto-created)
├── requirements.txt              # Python dependencies (works with CPU or CUDA)
├── requirements_py313.txt        # Adjusted requirements for Python 3.13+
├── Dockerfile                    # Docker configuration for containerized deployment
├── render.yaml / render.toml     # Render.com deployment configuration
└── STCA_Net_System_Overview_Diagrams.md  # Mermaid architecture diagrams
```

> **Note:** Model weights (`.pt`), datasets, and the `venv/` directory are excluded from the repository via `.gitignore`.

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10 – 3.13**
- pip

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Ramanakumarreddy/STCA-Net-Project.git
cd STCA-Net-Project

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Python 3.13 users:** Use `pip install -r requirements_py313.txt` instead.

### 3. Run the Web App

If pre-trained weights (`models/stca_net_weights.pt`) are already present:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser to access the Deepfake Scanner Dashboard.

> If no weights are found, the app starts with a randomly initialized model (for UI testing only).

---

## 🧠 Model Architecture

### Overview

STCA-Net processes an input image or video clip through a six-stage pipeline:

```
Input (Image or Video Frames)
        │
        ▼
┌─────────────────────────────┐
│  1. Spatial Extraction      │  MobileNetV3-Small CNN extracts local feature maps
│     (B×T, 3, 384, 384)      │  → (B×T, 576, 12, 12)
└─────────────┬───────────────┘
              │
        ┌─────▼──────┐
        │  Conv 1×1  │  Channel projection: 576 → 256 (d_model)
        └─────┬──────┘
              │
┌─────────────▼───────────────┐
│  2. Global Context Encoder  │  2-layer Transformer Encoder with CLS token
│     (144 patches + CLS)     │  Adds learned positional embeddings
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  3. Frequency Domain Branch │  FFT magnitude spectrum → log scaling
│     (B×T, 3, 384, 384)      │  → 2×Conv + BatchNorm → Linear → (B×T, 256)
└─────────────┬───────────────┘
              │  (summed into CLS token)
┌─────────────▼───────────────┐
│  4. Cross-Attention Fusion  │  CLS (global+freq) queries the CNN spatial maps
│                             │  Merges global anomalies with local pixel features
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  5. Temporal Encoder        │  1-layer Transformer over the T-frame sequence
│     (B, T, 256)             │  → mean-pooled to (B, 256)
└─────────────┬───────────────┘
              │
┌─────────────▼───────────────┐
│  6. Classification Head     │  Linear(256→128) → LayerNorm → ReLU → Dropout
│                             │  → Linear(128→2) → [REAL, FAKE]
└─────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Rationale |
|---|---|---|
| CNN Backbone | MobileNetV3-Small (pretrained ImageNet) | Lightweight; captures fine-grained texture/blending artifacts |
| Patch Size | 12×12 = 144 spatial patches | Balances resolution against transformer sequence length |
| d_model | 256 | Keeps parameter count under 10 MB |
| Frequency Branch | FFT (in model) + DCT (in prediction pipeline) | FFT learns frequency filters end-to-end; DCT provides interpretable scores |
| Cross-Attention | CLS as Query, CNN maps as K/V | Forces the global context to localize back to suspicious regions |
| Temporal Encoder | 1-layer Transformer, mean-pooled | Temporal consistency across frames without heavy overhead |

### Parameter Count

```bash
python models/stca_net.py
# Output: STCA-Net Total Trainable Parameters: ~9.6M
```

---

## 🎓 Training

### Dataset: Celeb-DF-v2

The model was trained on [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics), which contains:
- **Celeb-real/**: 590 real celebrity videos
- **Celeb-synthesis/**: 5,639 high-quality deepfake videos
- **YouTube-real/**: 300 real YouTube videos

### Step 1: Prepare the Dataset

Extract face frames from the raw Celeb-DF-v2 videos into the required `real/` and `fake/` folder structure:

```bash
python process_raw_videos.py \
    --raw-real-dir Celeb-DF-v2/Celeb-real \
    --raw-fake-dir Celeb-DF-v2/Celeb-synthesis \
    --out-real-dir dataset/benchmark_data/real \
    --out-fake-dir dataset/benchmark_data/fake \
    --frames-per-video 15
```

### Step 2a: Standard Training

For datasets that fit in memory (or when using a GPU/Colab):

```bash
python train_stca_net.py \
    --dataset dataset/benchmark_data \
    --epochs 15 \
    --batch-size 4 \
    --seq-len 5 \
    --samples 10000 \
    --lr 0.0001 \
    --unfreeze-layers 3 \
    --scheduler cosine \
    --patience 5
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dataset/140k` | Path to folder with `real/` and `fake/` subdirectories |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `4` | Mini-batch size (lower for CPU) |
| `--seq-len` | `5` | Frames per video sequence |
| `--samples` | `10000` | Max sequences to load (0 = all) |
| `--lr` | `0.0001` | AdamW learning rate |
| `--unfreeze-layers` | `3` | Number of MobileNet tail layers to fine-tune |
| `--label-smoothing` | `0.1` | Focal loss label smoothing factor |
| `--scheduler` | `cosine` | `cosine` (CosineAnnealingWarmRestarts) or `plateau` |
| `--patience` | `5` | Early stopping patience in epochs |
| `--resume-weights` | `None` | Path to `.pt` file to resume from |

Weights are saved to `models/stca_net_weights.pt` when validation accuracy improves.

### Step 2b: Chunk-Based Training (CPU / Low-RAM)

For training on the full Celeb-DF-v2 dataset on a machine with limited RAM or storage, use the progressive chunk trainer:

```bash
python train_in_chunks.py
# or resume from a specific chunk:
python train_in_chunks.py --start-chunk 5
```

**How it works:**
1. Splits the dataset into chunks of 20 videos per class.
2. Extracts 5 sharp frames per video in the current chunk.
3. Injects frames from a **replay buffer** (up to 40/class) so the model retains memory of earlier data.
4. Trains for 5 epochs on the combined chunk + replay data.
5. Updates the replay buffer with a random sample of new frames.
6. Wipes the chunk data folder to reclaim disk space.
7. Repeats for all chunks.

**Tunable constants** (top of `train_in_chunks.py`):

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 20 | Videos per class per chunk |
| `FRAMES_PER_VIDEO` | 5 | Frames extracted per video |
| `EPOCHS_PER_CHUNK` | 5 | Epochs trained per chunk |
| `REPLAY_BUFFER_CAP` | 200 | Max images per class in the replay buffer |
| `REPLAY_MIX_COUNT` | 40 | Images per class injected from replay each chunk |

### Training on Google Colab

Use the provided notebook `STCA_Net_Training_Colab.ipynb` for GPU-accelerated training. Mount your Google Drive containing the Celeb-DF-v2 dataset and project files, then run all cells.

---

## 📊 Inference Pipeline

### Image Prediction

When an image is submitted, `predict_image()` in `utils/prediction.py` runs the following cascade:

```
1. AI Signature Check        → filename / EXIF metadata scan
   ↓ (if signature found: override to FAKE at 95%)
2. Non-Photographic Check    → saturation, edge density, texture variance
   ↓ (if non-photo: attach warning; prediction still runs)
3. Face Detection            → Haar Cascade; falls back to center crop
4. Neural Network            → STCA-Net forward pass on face-cropped image
5. DCT Frequency Analysis    → band-energy score (0=real, 1=AI-generated)
6. Score Fusion
   Combined Fake  = (NN_fake × 0.70) + (freq_score × 0.30)
   Combined Real  = (NN_real × 0.70) + ((1-freq_score) × 0.30)
```

**Output dict:**

```python
{
    'prediction':         'REAL' | 'FAKE',
    'confidence':         float,          # % of winning class
    'fake_probability':   float,          # % combined
    'real_probability':   float,          # % combined
    'nn_fake_probability': float,         # % NN only
    'nn_real_probability': float,         # % NN only
    'frequency_score':    float,          # % DCT AI-likeness
    'face_detected':      bool,
    'signature_found':    bool,
    'signature_reason':   str,
    'is_non_photographic': bool,
    'warning':            str,            # only present if non-photographic
    'attention_map':      np.ndarray,     # shape (1, T, 1, 144)
    'processing_time':    float,          # seconds (added by app.py)
    'model_used':         str,
}
```

### Video Prediction

For videos, `extract_frames_from_video()` first runs **smart frame selection**:
- Divides the video into `max_frames` equal intervals.
- Within each interval, samples up to 5 candidate frames and selects the sharpest one using **Variance of Laplacian**.
- Runs Haar Cascade face detection and crops to the largest face (+ 20% margin).

The resulting list of face-cropped PIL Images is then passed to `predict_video_frames()`:
- Stacks frames into a `(1, T, C, H, W)` tensor for the temporal STCA-Net pass.
- Also runs each frame individually for **per-frame score timelines**.
- Combines NN and DCT scores with the same 70/30 weighting as images.

---

## 🔬 Evaluation

### Quick Validation

Run against the last processed `chunk_data` or any `benchmark_data` folder:

```bash
python validate_model.py
```

### Comprehensive End-to-End Evaluation

Randomly samples videos from across the **entire Celeb-DF-v2 dataset** (not just the training split):

```bash
python eval_comprehensive.py
```

Outputs accuracy, precision, recall, F1, and a pass/fail verdict:
- ≥ 85% accuracy → **EXCELLENT** (production ready)
- ≥ 75% accuracy → **GOOD** (more training recommended)
- ≥ 60% accuracy → **FAIR** (significant training needed)
- < 60% accuracy → **POOR** (has not generalised)

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t stca-net .

# Run the container
docker run -p 10000:10000 stca-net
```

The Dockerfile uses Python 3.10-slim, installs all system dependencies for OpenCV headless rendering, and starts `server.py` on port 10000.

For **Render.com** cloud deployment, `render.yaml` and `render.toml` are pre-configured.

---

## ⚙️ Configuration Reference

### Prediction Thresholds (`utils/prediction.py`)

| Parameter | Value | Notes |
|---|---|---|
| NN weight | 0.70 | Weight of neural network score in final decision |
| Freq weight | 0.30 | Weight of DCT frequency score |
| Signature override | 0.95 fake | Applied when AI metadata/filename signature found |
| DCT high-freq threshold | < 0.05 → +0.40 score | Low high-frequency energy is suspicious |
| DCT spectral slope | < -0.08 → +0.30 score | Steep decay indicates smooth AI textures |
| DCT low-energy concentration | > 0.85 → +0.30 score | Over-concentrated low frequencies |
| Non-photo anime score threshold | ≥ 0.50 | Image flagged as non-photographic |

### Flask App (`app.py`)

| Config Key | Value |
|---|---|
| `MAX_CONTENT_LENGTH` | 500 MB |
| `UPLOAD_FOLDER` | `Uploaded_Files/` |
| `VIDEO_UPLOAD_FOLDER` | `Uploaded_Files/videos/` |
| `IMAGE_UPLOAD_FOLDER` | `Uploaded_Files/images/` |
| Supported video formats | `.mp4`, `.avi`, `.mov` |
| Supported image formats | `.jpg`, `.jpeg`, `.png` |

---

## 📄 License

MIT License. Free to use, fork, and modify for research or educational purposes.

---

## 🙏 Acknowledgements

- **MobileNetV3** — Howard et al., *Searching for MobileNetV3*, 2019
- **Vision Transformer (ViT)** — Dosovitskiy et al., *An Image is Worth 16×16 Words*, 2021
- **Celeb-DF-v2** — Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020
- **Focal Loss** — Lin et al., *Focal Loss for Dense Object Detection*, 2017
