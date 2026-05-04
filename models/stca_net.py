import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft

class STCANet(nn.Module):
    """
    Spatio-Temporal Cross-Attention Network (STCA-Net) for Deepfake Detection.

    A lightweight hybrid architecture that combines CNN spatial features,
    Transformer global context, and frequency-domain analysis to detect
    pixel-level and structural artifacts in deepfake images and videos.

    Architecture Stages:
        1. Spatial Extraction  — MobileNetV3-Small (pretrained, partially frozen)
                                 extracts fine-grained local texture/blending features.
        2. Global Context      — Shallow Vision Transformer (2-layer, 8-head)
                                 over 144 spatial patches + a learnable CLS token
                                 captures long-range structural inconsistencies.
        3. Frequency Domain    — 2D FFT magnitude spectrum is log-scaled and fed
                                 through a small CNN to capture spectral fingerprints
                                 of AI-generated content (GANs, diffusion models).
        4. Cross-Attention     — The CLS token (augmented with freq features) attends
                                 back to the CNN spatial patch map, grounding the
                                 global anomaly signal to specific face regions.
        5. Temporal Modeling   — For video input, a 1-layer Transformer processes
                                 the per-frame fused representations as a sequence,
                                 then mean-pools over time.
        6. Classification Head — Linear → LayerNorm → ReLU → Dropout → Linear(2).

    Args:
        num_classes (int): Number of output classes. Default: 2 (Fake=0, Real=1).
        d_model (int): Transformer hidden dimension / channel projection size. Default: 256.
        nhead (int): Number of attention heads in all Transformer layers. Default: 8.
        num_encoder_layers (int): Number of layers in the global context encoder. Default: 2.

    Input:
        x: Tensor of shape (B, C, H, W) for a single image, or
               (B, T, C, H, W) for a video clip of T frames.
           Images are expected at 384×384 resolution, ImageNet-normalized.

    Output:
        output (Tensor):       Logits of shape (B, 2).
        attn_weights (Tensor): Cross-attention weights of shape (B, T, 1, 144),
                               where 144 = 12×12 spatial patches. Useful for
                               generating saliency / attention maps.
    """
    def __init__(self, num_classes=2, d_model=256, nhead=8, num_encoder_layers=2):
        super(STCANet, self).__init__()
        
        # ── Stage 1: Local Spatial Extractor (MobileNetV3-Small) ─────────────
        # Pre-trained on ImageNet so it starts with rich texture and edge detectors.
        # We only use the 'features' portion (no classifier head).
        # Output shape for a 384×384 input: (B*T, 576, 12, 12)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.spatial_extractor = mobilenet.features
        
        # 1×1 convolution to project CNN channels from 576 → d_model (256).
        # Keeps the spatial grid intact while aligning channel dim with the Transformer.
        self.conv_proj = nn.Conv2d(576, d_model, kernel_size=1)
        
        # ── Stage 2: Global Context Encoder (Shallow Vision Transformer) ─────
        # Flattening the 12×12 spatial map yields 144 tokens treated as a sequence.
        # A learnable positional embedding preserves spatial awareness.
        self.pos_embedding = nn.Parameter(torch.randn(1, 144, d_model))
        # CLS token aggregates global information across all 144 patches.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Small bottleneck to keep parameters low
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # ── Stage 3 (pre-requisite): Temporal Encoder ──────────────────────────
        # After cross-attention produces one fused vector per frame, this 1-layer
        # Transformer models consistency (or inconsistency) across T frames,
        # helping detect temporal artifacts in deepfake videos.
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        
        # ── Stage 4: Cross-Attention Fusion ─────────────────────────────────────
        # The CLS token (enriched with global context + frequency features) acts as
        # the Query. The 144 CNN spatial patches serve as Keys and Values.
        # This grounds the global anomaly signal in specific local face regions,
        # and produces interpretable attention heat-maps as a by-product.
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # ── Stage 3: Frequency Domain Extractor ──────────────────────────────
        # AI-generated images (GANs, Latent Diffusion) have distinct spectral
        # fingerprints: smoother high-frequency decay and concentrated low-frequency
        # energy. This small CNN processes the log-scaled FFT magnitude spectrum
        # to learn these spectral artifacts end-to-end.
        # Input: log(|FFT2(x)| + 1e-8), same shape as x: (B*T, 3, H, W)
        # Output: (B*T, d_model) frequency embedding
        self.freq_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model)
        )
        
        # ── Stage 6: Classification Head ──────────────────────────────────────
        # Takes the temporally-pooled fused representation (B, 256) and maps it
        # to class logits. LayerNorm stabilises training; Dropout prevents overfitting.
        # Output: (B, num_classes) — index 0 = Fake, index 1 = Real.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass for both image and video inputs.

        Args:
            x (Tensor): Either (B, C, H, W) for a single image batch, or
                        (B, T, C, H, W) for a video sequence batch.
                        Expects 384×384 resolution, ImageNet-normalized.

        Returns:
            output (Tensor):       Class logits of shape (B, 2).
                                   Index 0 = Fake probability, Index 1 = Real probability.
            attn_weights (Tensor): Cross-attention weights of shape (B, T, 1, 144).
                                   Represents which of the 144 spatial patches the model
                                   focused on. Useful for generating saliency maps.
        """
        # Unify image and video inputs: images are treated as single-frame videos.
        # (B, C, H, W) → (B, 1, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.size()

        # Merge batch and time dimensions for frame-level CNN processing.
        x_flat = x.view(B * T, C, H, W)  # (B*T, C, H, W)

        # ── Stage 1: Spatial Feature Extraction ───────────────────────────────
        # (B*T, 3, 384, 384) → (B*T, 576, 12, 12)
        spatial_features = self.spatial_extractor(x_flat)

        # Project channels: (B*T, 576, 12, 12) → (B*T, 256, 12, 12)
        proj_features = self.conv_proj(spatial_features)

        # Flatten spatial grid to sequence: (B*T, 256, 144) → (B*T, 144, 256)
        seq_features = proj_features.flatten(2).transpose(1, 2)

        # ── Stage 2: Global Context Encoding (Shallow ViT) ────────────────────
        # Add positional embeddings to preserve spatial awareness.
        seq_features = seq_features + self.pos_embedding  # (B*T, 144, 256)

        # Prepend the learnable CLS token. The Transformer will use this to
        # aggregate a global summary of the entire patch sequence.
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, 256)
        transformer_input = torch.cat((cls_tokens, seq_features), dim=1)  # (B*T, 145, 256)

        transformer_output = self.transformer_encoder(transformer_input)  # (B*T, 145, 256)

        # Extract only the CLS output as the global context summary.
        global_context = transformer_output[:, 0:1, :]  # (B*T, 1, 256)

        # ── Stage 3: Frequency Domain Injection ───────────────────────────────
        # Compute the 2D FFT magnitude spectrum of each frame.
        # fftshift centres the zero-frequency component for better convolution learning.
        freq_complex = torch.fft.fft2(x_flat, norm='ortho')
        freq_mag = torch.abs(freq_complex)                        # (B*T, 3, H, W)
        freq_mag = torch.fft.fftshift(freq_mag, dim=(-2, -1))    # centre zero-freq
        freq_mag = torch.log(freq_mag + 1e-8)                    # log-scale for stability

        # Extract frequency embedding and add it to the CLS global context.
        freq_feat = self.freq_extractor(freq_mag)   # (B*T, 256)
        freq_feat = freq_feat.unsqueeze(1)          # (B*T, 1, 256)
        global_context = global_context + freq_feat  # enrich CLS with spectral signal

        # ── Stage 4: Cross-Attention Fusion ───────────────────────────────────
        # The enriched CLS token (Query) attends to all 144 CNN spatial patches
        # (Keys & Values), grounding the global anomaly signal to local face regions.
        fused_features, attn_weights = self.cross_attention(
            query=global_context,   # (B*T, 1, 256)
            key=seq_features,       # (B*T, 144, 256)
            value=seq_features      # (B*T, 144, 256)
        )  # fused_features: (B*T, 1, 256) | attn_weights: (B*T, 1, 144)

        # ── Stage 5: Temporal Modeling ────────────────────────────────────────
        # Reshape back to (B, T, 256) to process the sequence of per-frame
        # fused representations across time.
        temporal_input = fused_features.squeeze(1).view(B, T, -1)  # (B, T, 256)
        temporal_features = self.temporal_encoder(temporal_input)   # (B, T, 256)

        # Mean-pool over the temporal dimension to get one video-level embedding.
        video_features = temporal_features.mean(dim=1)  # (B, 256)

        # ── Stage 6: Classification ───────────────────────────────────────────
        output = self.classifier(video_features)  # (B, 2)

        # Reshape attention weights to (B, T, 1, 144) for downstream saliency map use.
        if attn_weights is not None:
            _, _, num_patches = attn_weights.shape
            attn_weights = attn_weights.view(B, T, 1, num_patches)

        return output, attn_weights
        
    def get_parameter_count(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Smoke test: verifies model construction, parameter count, and output shapes.
    model = STCANet()
    print(f"STCA-Net Total Trainable Parameters: {model.get_parameter_count():,}")

    # ── Single image input ────────────────────────────────────────────────────
    # Expected: logits (1, 2), attn (1, 1, 1, 144)
    dummy_input_image = torch.randn(1, 3, 384, 384)
    output_img, attn_img = model(dummy_input_image)
    print(f"Single Image — Output: {output_img.shape} | Attention: {attn_img.shape}")

    # ── Video clip input (batch=1, 5 frames) ─────────────────────────────────
    # Expected: logits (1, 2), attn (1, 5, 1, 144)
    dummy_input_video = torch.randn(1, 5, 3, 384, 384)
    output_vid, attn_vid = model(dummy_input_video)
    print(f"Video Clip   — Output: {output_vid.shape} | Attention: {attn_vid.shape}")
