# Dataset Setup Guide for DeepFake Detection

## 📁 Folder Structure

```
Admin/datasets/
├── Celeb-DF/
│   ├── Celeb-real/        ← Real celebrity videos
│   ├── Celeb-synthesis/   ← Deepfake videos
│   └── YouTube-real/      ← Real YouTube videos
└── FF++/
    ├── real/              ← Original videos
    ├── Deepfakes/         ← Deepfake manipulated
    ├── Face2Face/         ← Face reenactment
    ├── FaceSwap/          ← Face swap
    └── NeuralTextures/    ← Neural texture manipulation
```

## 📥 Download Links

### Option 1: Celeb-DF Dataset (Recommended)
- **Kaggle:** https://www.kaggle.com/datasets/reubensuju/celebdf
- **Size:** ~5GB
- Download and extract into `Admin/datasets/Celeb-DF/` folder

### Option 2: FaceForensics++ 
- **Kaggle:** https://www.kaggle.com/datasets/sorokin/faceforensics
- **Size:** ~10GB+
- Download and extract into `Admin/datasets/FF++/` folder

### Option 3: Quick Start - 140K Faces (Images Only)
- **Kaggle:** https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **Size:** ~1GB
- Good for quick testing with images

## 📋 File Requirements

- **Format:** `.mp4` video files
- **Content:** Videos must contain visible faces
- **Minimum:** At least 50 real + 50 fake videos recommended

## 🚀 After Downloading

1. Extract dataset to the appropriate folder
2. Ensure videos are `.mp4` format
3. Run training: `python advanced_training.py`

## ⚠️ Important Notes

- Keep folder names exactly as shown above
- Labels are assigned automatically based on folder:
  - `Celeb-real/`, `YouTube-real/`, `FF++/real/` → Label = 1 (Real)
  - Other folders → Label = 0 (Fake)
