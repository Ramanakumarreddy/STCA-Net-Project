# TrueVision - DeepFake Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

An AI-powered DeepFake Detection Web Application using Deep Learning (EfficientNet + LSTM with Attention) that predicts whether a video or image is **FAKE** or **REAL** with confidence scores.

## ✨ Features

- **Video Detection** - Analyze videos for deepfake manipulation
- **Image Detection** - Detect AI-generated or manipulated images
- **95%+ Accuracy** - State-of-the-art detection performance
- **Fast Processing** - Results in under 5 seconds
- **Clean Modern UI** - Simple, professional interface
- **Secure & Private** - Files deleted after analysis
- **Admin Dashboard** - User management and training controls

## 🖥️ Screenshots

The application features a clean, modern design with:
- Responsive navigation
- Intuitive upload interface
- Clear result visualization with confidence charts
- Mobile-friendly layout

## 🚀 Quick Start

### Prerequisites

- Python 3.10.x or 3.13.x
- pip (Python package installer)
- Windows OS (for provided wheel files) or Linux/Mac

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mounish7028/Deepfake-Face-Detection.git
   cd dfd
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   
   For Python 3.10:
   ```bash
   pip install dlib-19.22.99-cp310-cp310-win_amd64.whl  # Windows only
   pip install -r requirements.txt
   ```
   
   For Python 3.13:
   ```bash
   pip install -r requirements_py313.txt
   ```

4. **Run the application**
   ```bash
   python server.py
   ```

5. **Access the app** at http://localhost:5000

## 📁 Project Structure

```
dfd/
├── model/                     # Model weights (df_model.pt)
├── static/
│   ├── simple.css            # Main design system
│   ├── index.css             # Additional styles
│   └── react/                # React components
├── templates/                 # HTML templates
│   ├── home.html             # Landing page
│   ├── detect.html           # Video detection
│   ├── image.html            # Image detection
│   ├── login.html            # User login
│   ├── signup.html           # User registration
│   └── admin.html            # Admin dashboard
├── Admin/datasets/           # Training datasets
├── Uploaded_Files/           # Temporary upload storage
├── server.py                 # Flask application
├── models.py                 # Database models
├── advanced_training.py      # Model training script
├── performance_monitor.py    # Performance tracking
├── requirements.txt          # Python 3.10 dependencies
└── requirements_py313.txt    # Python 3.13 dependencies
```

## 🧠 Model Architecture

The detection system uses an advanced deep learning architecture:

```
EfficientNet-B2 Backbone (Feature Extraction)
            ↓
   Spatial Attention Layer
            ↓
  Bidirectional LSTM (2 layers)
            ↓
   Multi-Head Temporal Attention
            ↓
   Dual-Head Classifier
   ├── Main: REAL/FAKE
   └── Synthetic: AI-generated detection
```

### Key Features
- **EfficientNet-B2** backbone for efficient feature extraction
- **Bidirectional LSTM** for temporal sequence modeling
- **Attention Mechanisms** for focusing on relevant features
- **Synthetic Fingerprint Detection** for GAN-generated content
- **Mixed Precision Training** for faster training

## 📊 Performance

| Metric | Value |
|--------|-------|
| Detection Accuracy | 95%+ |
| Image Processing | < 5 seconds |
| Video Processing | < 5 seconds |
| False Positive Rate | < 10% |

### Supported Datasets
- Celeb-DF
- FaceForensics++
- YouTube-Real

## 🎨 Design System

The application uses a clean, modern design with:

- **Typography**: Inter font family
- **Colors**: Blue primary (#3b82f6), neutral grays
- **Components**: Cards, buttons, forms with consistent styling
- **Responsive**: Mobile-first design approach

## 🔧 Configuration

### Environment Variables
```bash
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///instance/users.db
```

### Model Training
```bash
python advanced_training.py
```

### Performance Testing
```bash
python test_model_performance.py
```

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/detect` | GET/POST | Video detection |
| `/image` | GET/POST | Image detection |
| `/login` | GET/POST | User login |
| `/signup` | GET/POST | User registration |
| `/admin` | GET | Admin dashboard |

## 🐳 Docker

```bash
docker build -t deepfake-detection .
docker run -p 5000:5000 deepfake-detection
```

## 📝 Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `df_model.pt` exists in the `model/` directory
   - Check PyTorch version compatibility

2. **Face detection failing**
   - Ensure MediaPipe is installed correctly
   - Check that uploaded videos contain visible faces

3. **Slow processing**
   - GPU acceleration requires CUDA-compatible PyTorch
   - Reduce frame count for faster analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Celeb-DF Dataset](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

---

**Built with ❤️ for combating misinformation**
