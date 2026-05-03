# STCA-Net Project — System Overview & Use Case Diagrams

> **Project:** STCA-Net (Spatio-Temporal Convolutional Attention Network)  
> **Purpose:** Efficient Video Action Recognition using Deep Learning  
> **Backbone:** R(2+1)D Network + STCA Attention Module  
> **Benchmark Datasets:** UCF-101 | HMDB-51

---

## How to Preview These Diagrams

All diagrams below use **Mermaid** syntax. To render them:

| Tool | Steps |
|------|-------|
| **VS Code** | Install the *Mermaid Preview* extension → open `.md` → `Ctrl+Shift+P` → "Mermaid: Preview" |
| **GitHub** | Push this `.md` file — GitHub auto-renders Mermaid blocks |
| **Online** | Paste any block at [mermaid.live](https://mermaid.live) |
| **Obsidian** | Enable "Mermaid" in settings → preview mode renders automatically |
| **Notion** | Use `/code` block → select `Mermaid` as language |

---

## 1. Use Case Diagram

```mermaid
%%{init: {"theme": "default"}}%%
graph LR
    subgraph Actors
        U1["👤 Researcher"]
        U2["👤 Developer"]
        U3["👤 End User / Application"]
    end

    subgraph STCA_Net_System["STCA-Net System"]
        UC1["Load & Preprocess Video"]
        UC2["Extract Spatial Features"]
        UC3["Extract Temporal Features"]
        UC4["Apply STCA Attention Module"]
        UC5["Perform Action Classification"]
        UC6["Evaluate Model Performance"]
        UC7["Train / Fine-tune Model"]
        UC8["Export / Deploy Model"]
        UC9["Benchmark on UCF-101 / HMDB-51"]
    end

    U1 --> UC7
    U1 --> UC6
    U1 --> UC9

    U2 --> UC1
    U2 --> UC7
    U2 --> UC8

    U3 --> UC1
    U3 --> UC5
    U3 --> UC8

    UC1 --> UC2
    UC1 --> UC3
    UC2 --> UC4
    UC3 --> UC4
    UC4 --> UC5
    UC5 --> UC6
    UC7 --> UC9
```

---

## 2. System Architecture Overview

```mermaid
flowchart TD
    A["🎥 Input Video Clip\n(RGB frames sequence)"]

    subgraph Backbone["R(2+1)D Backbone Network"]
        B["2D Spatial Convolution\n(per frame)"]
        C["1D Temporal Convolution\n(across frames)"]
    end

    subgraph STCA_Module["STCA Attention Module"]
        D["Spatial Attention\n🗺️ WHERE to look"]
        E["Temporal Attention\n⏱️ WHEN to focus"]
        F["Feature Fusion\n(Spatial × Temporal)"]
    end

    subgraph Classifier["Classification Head"]
        G["Global Average Pooling"]
        H["Fully Connected Layer"]
        I["Softmax Output\n(Action Classes)"]
    end

    A --> B
    A --> C
    B --> D
    C --> E
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I

    style Backbone fill:#dbeafe,stroke:#3b82f6
    style STCA_Module fill:#dcfce7,stroke:#22c55e
    style Classifier fill:#fef9c3,stroke:#eab308
```

---

## 3. Data Flow Diagram

```mermaid
flowchart LR
    V["🎞️ Raw Video\n(.mp4 / .avi)"]
    P["📦 Preprocessing\n• Frame Sampling\n• Resize & Normalize\n• Clip Formation"]
    FE["🧠 Feature Extractor\nR(2+1)D Layers"]
    ATT["🔍 STCA Module\n• Spatial Branch\n• Temporal Branch"]
    CLS["🏷️ Classifier\nFC + Softmax"]
    OUT["✅ Predicted Action\ne.g. Running, Jumping..."]

    V --> P --> FE --> ATT --> CLS --> OUT

    style V fill:#f3e8ff,stroke:#a855f7
    style P fill:#ffe4e6,stroke:#f43f5e
    style FE fill:#dbeafe,stroke:#3b82f6
    style ATT fill:#dcfce7,stroke:#22c55e
    style CLS fill:#fef9c3,stroke:#eab308
    style OUT fill:#ccfbf1,stroke:#14b8a6
```

---

## 4. Component Diagram

```mermaid
flowchart TD
    subgraph Input_Pipeline["Input Pipeline"]
        IP1["Video Loader"]
        IP2["Frame Sampler"]
        IP3["Data Augmentation"]
        IP4["DataLoader / Batch Builder"]
    end

    subgraph Model["STCA-Net Model"]
        M1["R(2+1)D Stem\n(Conv3D decomposed)"]
        M2["Residual Blocks\n(Layer1 → Layer4)"]
        M3["STCA Module\n(Spatial + Temporal Attention)"]
        M4["Global Average Pool"]
        M5["FC Classification Head"]
    end

    subgraph Training["Training Pipeline"]
        T1["Loss Function\n(CrossEntropyLoss)"]
        T2["Optimizer\n(SGD / Adam)"]
        T3["LR Scheduler"]
        T4["Checkpoint Saver"]
    end

    subgraph Evaluation["Evaluation"]
        E1["Top-1 Accuracy"]
        E2["Top-5 Accuracy"]
        E3["Confusion Matrix"]
        E4["FLOPs / Params Report"]
    end

    Input_Pipeline --> Model
    Model --> Training
    Training --> Evaluation

    IP1 --> IP2 --> IP3 --> IP4
    M1 --> M2 --> M3 --> M4 --> M5
    T1 --> T2 --> T3 --> T4
    E1 & E2 & E3 & E4

    style Input_Pipeline fill:#ffe4e6,stroke:#f43f5e
    style Model fill:#dbeafe,stroke:#3b82f6
    style Training fill:#fef9c3,stroke:#eab308
    style Evaluation fill:#dcfce7,stroke:#22c55e
```

---

## 5. Training Lifecycle Sequence

```mermaid
sequenceDiagram
    participant D as Dataset (UCF-101/HMDB-51)
    participant L as DataLoader
    participant M as STCA-Net Model
    participant LF as Loss Function
    participant O as Optimizer
    participant E as Evaluator

    loop Each Epoch
        D ->> L: Provide video clips + labels
        L ->> M: Forward batch (clips)
        M ->> M: Spatial Conv (R2D)
        M ->> M: Temporal Conv (1D)
        M ->> M: STCA Attention
        M ->> LF: Predicted logits
        LF ->> O: Compute CrossEntropy Loss
        O ->> M: Backprop + Update weights
    end

    M ->> E: Run on validation split
    E -->> M: Top-1 / Top-5 Accuracy
```

---

## 6. STCA Module Internal Design

```mermaid
flowchart LR
    IN["Feature Map\n(B × C × T × H × W)"]

    subgraph Spatial_Branch["🗺️ Spatial Branch"]
        SA1["Avg Pool → Temporal dim"]
        SA2["Lightweight 2D Conv"]
        SA3["Sigmoid → Spatial Mask"]
    end

    subgraph Temporal_Branch["⏱️ Temporal Branch"]
        TA1["Avg Pool → Spatial dims"]
        TA2["Lightweight 1D Conv"]
        TA3["Sigmoid → Temporal Mask"]
    end

    FUSE["⊗ Element-wise Multiply\n(Feature × Spatial Mask × Temporal Mask)"]
    OUT["Attended Feature Map\n(B × C × T × H × W)"]

    IN --> SA1 --> SA2 --> SA3 --> FUSE
    IN --> TA1 --> TA2 --> TA3 --> FUSE
    IN --> FUSE
    FUSE --> OUT

    style Spatial_Branch fill:#dbeafe,stroke:#3b82f6
    style Temporal_Branch fill:#dcfce7,stroke:#22c55e
```

---

## 7.  System Entity Relationship (Conceptual)

```mermaid
erDiagram
    VIDEO {
        string video_id
        string file_path
        int total_frames
        float fps
        string dataset_split
    }
    CLIP {
        string clip_id
        string video_id
        int start_frame
        int end_frame
        int num_frames
    }
    ACTION_CLASS {
        int class_id
        string class_name
        string dataset
    }
    MODEL_RUN {
        string run_id
        string model_version
        float top1_accuracy
        float top5_accuracy
        int epochs_trained
    }
    PREDICTION {
        string prediction_id
        string clip_id
        int predicted_class
        float confidence
        string run_id
    }

    VIDEO ||--o{ CLIP : "sampled into"
    CLIP ||--|| ACTION_CLASS : "labelled as"
    MODEL_RUN ||--o{ PREDICTION : "generates"
    CLIP ||--|| PREDICTION : "evaluated by"
```

