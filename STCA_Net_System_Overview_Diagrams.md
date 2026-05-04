# STCA-Net — System Architecture & Diagrams

> **Project:** STCA-Net (Spatio-Temporal Cross-Attention Network)
> **Purpose:** Hybrid deepfake detection for images and video using CNN + Transformer + Frequency analysis
> **Dataset:** Celeb-DF-v2 (deepfake forensics benchmark)

---

## How to Preview These Diagrams

All diagrams use **Mermaid** syntax. To render them:

| Tool | Steps |
|------|-------|
| **VS Code** | Install the *Mermaid Preview* extension → open `.md` → `Ctrl+Shift+P` → "Mermaid: Preview" |
| **GitHub** | Push this `.md` file — GitHub auto-renders Mermaid blocks |
| **Online** | Paste any block at [mermaid.live](https://mermaid.live) |
| **Obsidian** | Enable "Mermaid" in settings → preview mode renders automatically |

---

## 1. Use Case Diagram

```mermaid
%%{init: {"theme": "default"}}%%
graph LR
    subgraph Actors
        U1["👤 Researcher / Developer"]
        U2["👤 End User"]
        U3["🖥️ Web Browser"]
    end

    subgraph STCA_Net_System["STCA-Net Deepfake Detection System"]
        UC1["Upload Image or Video"]
        UC2["Face Detection & Cropping"]
        UC3["Neural Network Inference"]
        UC4["DCT Frequency Analysis"]
        UC5["AI Signature Check"]
        UC6["Non-Photo Detection"]
        UC7["Score Fusion & Decision"]
        UC8["View Dashboard Results"]
        UC9["Train / Fine-tune Model"]
        UC10["Evaluate Model Performance"]
    end

    U2 --> UC1
    U3 --> UC1
    U2 --> UC8

    U1 --> UC9
    U1 --> UC10

    UC1 --> UC2
    UC1 --> UC5
    UC1 --> UC6
    UC2 --> UC3
    UC2 --> UC4
    UC3 --> UC7
    UC4 --> UC7
    UC5 --> UC7
    UC6 --> UC7
    UC7 --> UC8
```

---

## 2. Full Inference Pipeline — Image

```mermaid
flowchart TD
    A["📷 Upload Image\n(.jpg / .png)"]

    subgraph PreScreen["Pre-screening Checks"]
        B["🔍 AI Signature Check\n(filename / EXIF metadata)"]
        C["🎨 Non-Photo Detection\n(saturation, edges, texture)"]
    end

    subgraph FaceExtract["Face Extraction (Haar Cascade)"]
        D{"Face\nDetected?"}
        D -- Yes --> E["Crop Largest Face\n(+20% margin)"]
        D -- No  --> F["Center Crop\n(fallback)"]
    end

    subgraph STCANet["STCA-Net Forward Pass"]
        G["Preprocess\n384×384, ImageNet Normalize"]
        H["MobileNetV3-Small\n→ (B, 576, 12, 12)"]
        I["Conv 1×1 Projection\n→ (B, 256, 12, 12)"]
        J["Flatten → 144 patches\n+ CLS token + Positional Embed"]
        K["Transformer Encoder\n2 layers, 8 heads"]
        L["FFT Magnitude Branch\n→ log-scaled freq features"]
        M["CLS + Freq Fusion"]
        N["Cross-Attention\nCLS queries CNN spatial maps"]
        O["Temporal Encoder\n(single-frame → size-1 seq)"]
        P["Classification Head\nLinear → Softmax → [FAKE, REAL]"]
    end

    subgraph FreqAnalysis["DCT Frequency Analysis"]
        Q["Grayscale → 256×256 DCT"]
        R["Band Energy:\nLow / Mid / High"]
        S["Spectral Decay Slope"]
        T["Frequency Score 0→1\n(0=real, 1=AI-generated)"]
    end

    subgraph Fusion["Score Fusion"]
        U["Combined Fake = NN_fake×0.70 + Freq×0.30"]
        V["Normalize → Final Probabilities"]
    end

    W["📊 Result: REAL / FAKE\n+ Confidence + Per-Component Scores"]

    A --> B
    A --> C
    A --> D
    E --> G
    F --> G
    G --> H --> I --> J --> K
    G --> L
    L --> M
    K --> M
    M --> N
    N --> O --> P

    A --> Q --> R --> S --> T

    P --> U
    T --> U
    B -- "Signature Found" --> V
    U --> V --> W

    style PreScreen fill:#fef3c7,stroke:#f59e0b
    style FaceExtract fill:#ede9fe,stroke:#8b5cf6
    style STCANet fill:#dbeafe,stroke:#3b82f6
    style FreqAnalysis fill:#dcfce7,stroke:#22c55e
    style Fusion fill:#fce7f3,stroke:#ec4899
```

---

## 3. STCA-Net Model Architecture

```mermaid
flowchart TD
    IN["Input Tensor\n(B, T, 3, 384, 384)\nor (B, 3, 384, 384)"]

    subgraph Spatial["Stage 1 — Spatial Extraction"]
        S1["MobileNetV3-Small\nfeatures extractor"]
        S2["Output: (B×T, 576, 12, 12)"]
        S3["Conv2d 1×1\nProject: 576 → 256 (d_model)"]
        S4["Flatten spatial: (B×T, 144, 256)"]
    end

    subgraph GlobalEnc["Stage 2 — Global Context Encoder"]
        G1["Add Learned Positional Embedding\n(1, 144, 256)"]
        G2["Prepend CLS Token\n→ (B×T, 145, 256)"]
        G3["TransformerEncoder\n2 layers | 8 heads | FF=512 | dropout=0.1"]
        G4["Extract CLS Output\n→ (B×T, 1, 256)"]
    end

    subgraph FreqBranch["Stage 3 — Frequency Domain"]
        F1["FFT2 on x_flat → magnitude"]
        F2["fftshift + log(mag + 1e-8)"]
        F3["freq_extractor:\nConv(3→32)→BN→ReLU→Conv(32→64)→BN→ReLU\n→AvgPool→Flatten→Linear(64→256)"]
        F4["Unsqueeze → (B×T, 1, 256)"]
    end

    subgraph CrossAttn["Stage 4 — Cross-Attention Fusion"]
        CA["MultiheadAttention\nQuery: CLS+Freq  |  Key,Value: CNN patches\n→ fused_features (B×T, 1, 256)"]
    end

    subgraph Temporal["Stage 5 — Temporal Modeling"]
        T1["Reshape: (B, T, 256)"]
        T2["TransformerEncoder\n1 layer | 8 heads | FF=512"]
        T3["Mean Pool over T\n→ (B, 256)"]
    end

    subgraph Clf["Stage 6 — Classification Head"]
        C1["Linear(256 → 128)"]
        C2["LayerNorm(128)"]
        C3["ReLU"]
        C4["Dropout(0.3)"]
        C5["Linear(128 → 2)"]
    end

    OUT["Output: logits (B, 2)\n[FAKE score, REAL score]\n+ Attention Weights (B, T, 1, 144)"]

    IN --> S1 --> S2 --> S3 --> S4
    S4 --> G1 --> G2 --> G3 --> G4
    IN --> F1 --> F2 --> F3 --> F4
    G4 --> CA
    F4 --> CA
    S4 --> CA
    CA --> T1 --> T2 --> T3
    T3 --> C1 --> C2 --> C3 --> C4 --> C5 --> OUT

    style Spatial fill:#dbeafe,stroke:#3b82f6
    style GlobalEnc fill:#ede9fe,stroke:#8b5cf6
    style FreqBranch fill:#dcfce7,stroke:#22c55e
    style CrossAttn fill:#fce7f3,stroke:#ec4899
    style Temporal fill:#fef3c7,stroke:#f59e0b
    style Clf fill:#fee2e2,stroke:#ef4444
```

---

## 4. Video Inference Pipeline

```mermaid
flowchart LR
    V["🎬 Upload Video\n(.mp4 / .avi / .mov)"]

    subgraph FrameExtract["Smart Frame Extraction (video_processing.py)"]
        FE1["Divide video into\n15 equal intervals"]
        FE2["Sample up to 5 frames\nper interval"]
        FE3["Select sharpest frame\n(Variance of Laplacian)"]
        FE4["Haar Cascade\nFace Detection + Crop"]
        FE5["Return: List of\n15 PIL Images (faces)"]
    end

    subgraph SequenceInfer["Sequence Inference (prediction.py)"]
        SI1["Preprocess each frame\n384×384, Normalize"]
        SI2["Stack: (1, 15, 3, 384, 384)"]
        SI3["STCA-Net forward\n(temporal sequence mode)"]
        SI4["Softmax → avg_nn_fake / avg_nn_real"]
        SI5["Per-frame individual\nforward passes (for timeline chart)"]
    end

    subgraph FreqVideo["Per-Frame Frequency Scores"]
        FV["DCT analysis on\neach frame independently\n→ avg_freq_score"]
    end

    subgraph FusionV["Score Fusion"]
        FU["Combined Fake = avg_nn_fake×0.70 + avg_freq×0.30\nNormalize → REAL/FAKE + Confidence"]
    end

    Result["📊 Result:\nPrediction | Confidence | Frames Analyzed\nNN Scores | Freq Score\nPer-frame Fake Scores | Per-frame Freq Scores"]

    V --> FE1 --> FE2 --> FE3 --> FE4 --> FE5
    FE5 --> SI1 --> SI2 --> SI3 --> SI4
    FE5 --> SI5
    FE5 --> FV
    SI4 --> FU
    FV --> FU
    FU --> Result

    style FrameExtract fill:#ede9fe,stroke:#8b5cf6
    style SequenceInfer fill:#dbeafe,stroke:#3b82f6
    style FreqVideo fill:#dcfce7,stroke:#22c55e
    style FusionV fill:#fce7f3,stroke:#ec4899
```

---

## 5. Training Lifecycle Sequence

```mermaid
sequenceDiagram
    participant DS as Dataset (real + fake)
    participant VP as VideoSequenceDataset
    participant DL as DataLoader + WeightedSampler
    participant M as STCA-Net Model
    participant FL as FocalLoss
    participant OPTIM as AdamW Optimizer
    participant SCHED as LR Scheduler
    participant ES as Early Stopping

    loop Each Epoch
        DS ->> VP: Load frame sequences (seq_len=5)
        VP ->> DL: Return (T, C, H, W) tensor + label
        DL ->> M: Forward batch (B, T, C, H, W)
        M ->> M: Spatial - Global - Freq - CrossAttn - Temporal
        M ->> FL: Logits + Ground Truth Labels
        FL ->> OPTIM: Focal Loss (gamma=2, label_smoothing=0.1)
        OPTIM ->> M: Backprop + clip_grad_norm + Weight Update
        M ->> ES: Validation Loss
        ES -->> OPTIM: Trigger LR step (cosine or plateau)
        ES -->> M: Save best weights if val_acc improves
        ES -->> SCHED: Stop if no improvement for patience epochs
    end

    M ->> M: Load best saved weights
    M ->> M: Compute Confusion Matrix on val set
    M ->> M: Print Accuracy, Precision, Recall, F1
    M ->> M: Save training_history.json
```

---

## 6. Chunk-Based Training with Replay Buffer

```mermaid
flowchart TD
    START["train_in_chunks.py\nScan Celeb-DF-v2"]
    SPLIT["Divide fake_videos into\nchunks of 20"]

    subgraph ChunkLoop["For each chunk (chunk_idx = 0 … N)"]
        C1["[1/4] Extract sharp frames\nfrom 20 real + 20 fake videos\n→ dataset/chunk_data/"]
        C2["[2/4] Inject Replay Buffer\nCopy 40 real + 40 fake frames\nfrom dataset/replay_buffer/"]
        C3["[3/4] Run train_stca_net.py\n--epochs 5 --resume-weights models/stca_net_weights.pt"]
        C4["[4/4] Update Replay Buffer\nSample new frames into buffer (cap=200/class)"]
        C5["Wipe chunk_data/\n(reclaim disk space)"]
    end

    DONE["All chunks complete\nFinal weights in models/stca_net_weights.pt"]

    START --> SPLIT --> C1 --> C2 --> C3 --> C4 --> C5
    C5 -->|Next chunk| C1
    C5 -->|All done| DONE

    style ChunkLoop fill:#fef3c7,stroke:#f59e0b
```

---

## 7. System Entity Relationship

```mermaid
erDiagram
    VIDEO {
        string video_id
        string file_path
        string class "real | fake"
        int total_frames
        float fps
        string source "Celeb-real | YouTube-real | Celeb-synthesis"
    }
    FRAME {
        string frame_id
        string video_id
        int frame_index
        float sharpness_score
        bool face_detected
        string file_path
    }
    PREDICTION {
        string prediction_id
        string input_type "image | video"
        string result "REAL | FAKE"
        float confidence
        float nn_fake_prob
        float nn_real_prob
        float frequency_score
        bool signature_found
        bool is_non_photographic
        float processing_time_s
    }
    MODEL_WEIGHTS {
        string weights_path
        int trainable_params
        float best_val_accuracy
        int total_epochs
        string training_dataset
    }

    VIDEO ||--o{ FRAME : "sampled into"
    FRAME ||--|| PREDICTION : "evaluated by"
    MODEL_WEIGHTS ||--o{ PREDICTION : "produces"
```

---

## 8. Flask Web Application Routes

```mermaid
flowchart LR
    Browser["🌐 Browser"]

    Browser --> R1["GET /\n→ index.html\n(Landing page)"]
    Browser --> R2["GET /detect\n→ detect.html\n(Video upload form)"]
    Browser --> R3["POST /detect\n→ extract_frames_from_video()\n→ predict_video_frames()\n→ detect.html (results)"]
    Browser --> R4["GET /image-detect\n→ image.html\n(Image upload form)"]
    Browser --> R5["POST /image-detect\n→ predict_image()\n→ image.html (results)"]

    R3 --> FS1["💾 Save video to\nUploaded_Files/videos/\n+ {filename}_prediction.json"]
    R5 --> FS2["💾 Save image to\nUploaded_Files/images/\n+ {filename}_prediction.json"]

    style R1 fill:#dcfce7,stroke:#22c55e
    style R2 fill:#dbeafe,stroke:#3b82f6
    style R3 fill:#dbeafe,stroke:#3b82f6
    style R4 fill:#ede9fe,stroke:#8b5cf6
    style R5 fill:#ede9fe,stroke:#8b5cf6
```
