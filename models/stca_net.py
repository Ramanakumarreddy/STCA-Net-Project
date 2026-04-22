import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft

class STCANet(nn.Module):
    """
    Spatio-Temporal Cross-Attention Hybrid Network (STCA-Net)
    A novel, lightweight architecture for Deepfake Detection.
    
    Architecture:
    1. Local Spatial Extractor: MobileNetV3-Small (Pretrained)
    2. Global Context Encoder: Shallow Vision Transformer (ViT)
    3. Fusion: Cross-Attention between Local and Global features
    """
    def __init__(self, num_classes=2, d_model=256, nhead=8, num_encoder_layers=2):
        super(STCANet, self).__init__()
        
        # 1. Local Spatial Extractor (MobileNetV3-Small)
        # We use a lightweight CNN to extract fine-grained pixel artifacts.
        # Output shape from feature extractor is typically (Batch, 576, 12, 12) for 384x384 input
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.spatial_extractor = mobilenet.features
        
        # Reduce the channel dimension from 576 to our d_model (256)
        self.conv_proj = nn.Conv2d(576, d_model, kernel_size=1)
        
        # 2. Global Context Encoder (Shallow Transformer)
        # We flatten the spatial dimensions (12x12 = 144 patches) and treat them as a sequence
        # We add a learned position embedding to retain spatial awareness
        self.pos_embedding = nn.Parameter(torch.randn(1, 144, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, # Keep bottleneck small
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Temporal Encoder
        # Processes sequences of global context tokens across time
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        
        # 4. Cross-Attention Fusion
        # We use the Transformer's CLS token to attend back to the original CNN spatial maps
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 5. Frequency Domain Extractor
        # Extracts features from the FFT magnitude spectrum of the input
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
        
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Base dimensions
        # x shape can be (B, C, H, W) for single image or (B, T, C, H, W) for video
        if x.dim() == 4:
            x = x.unsqueeze(1) # (B, 1, C, H, W)
            
        B, T, C, H, W = x.size()
        
        # Flatten B and T for spatial extraction
        x_flat = x.view(B * T, C, H, W)
        
        # Step 1: Spatial Extraction (CNN)
        # Input: (B*T, 3, 384, 384) -> Output: (B*T, 576, 12, 12)
        spatial_features = self.spatial_extractor(x_flat)
        
        # Project to d_model: (B*T, 576, 12, 12) -> (B*T, 256, 12, 12)
        proj_features = self.conv_proj(spatial_features)
        
        # Flatten for Transformer: (B*T, 256, 144) -> (B*T, 144, 256)
        seq_features = proj_features.flatten(2).transpose(1, 2)
        
        # Step 2: Global Context (Transformer)
        # Add position embeddings
        seq_features = seq_features + self.pos_embedding
        
        # Append CLS token: (B*T, 145, 256)
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        transformer_input = torch.cat((cls_tokens, seq_features), dim=1)
        
        # Pass through shallow Transformer
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Extract the processed CLS token (global context summary)
        global_context = transformer_output[:, 0:1, :] # (B*T, 1, 256)
        
        # Step 3: Frequency Domain Injection
        # Compute 2D FFT (Fast Fourier Transform) magnitude spectrum
        freq_complex = torch.fft.fft2(x_flat, norm='ortho')
        freq_mag = torch.abs(freq_complex)
        
        # Apply FFT shift and log scaling for better feature learning
        freq_mag = torch.fft.fftshift(freq_mag, dim=(-2, -1))
        freq_mag = torch.log(freq_mag + 1e-8)
        
        # Extract frequency features
        freq_feat = self.freq_extractor(freq_mag) # (B*T, 256)
        freq_feat = freq_feat.unsqueeze(1) # (B*T, 1, 256)
        
        # Fuse frequency-domain context into our spatial-global context before cross-attention
        global_context = global_context + freq_feat
        
        # Step 4: Cross-Attention Fusion
        # Query: Global Context (Transformer), Key/Value: Local Features (CNN)
        fused_features, attn_weights = self.cross_attention(
            query=global_context,
            key=seq_features,
            value=seq_features
        ) # fused_features: (B*T, 1, 256)
        
        # Step 5: Temporal Modeling
        # Reshape to (B, T, 256)
        temporal_input = fused_features.squeeze(1).view(B, T, -1)
        
        # Pass through temporal encoder
        temporal_features = self.temporal_encoder(temporal_input) # (B, T, 256)
        
        # Pool over time (average)
        video_features = temporal_features.mean(dim=1) # (B, 256)
        
        # Step 6: Classification
        output = self.classifier(video_features)
        
        # Reshape attention weights back to separate B and T for easier visualization downstream
        # attn_weights shape is (B*T, 1, 144)
        if attn_weights is not None:
            _, _, num_patches = attn_weights.shape
            attn_weights = attn_weights.view(B, T, 1, num_patches)
            
        return output, attn_weights
        
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick test to verify shapes and parameters
    model = STCANet()
    print(f"STCA-Net Total Trainable Parameters: {model.get_parameter_count():,}")
    
    # Dummy input single image (1 image, 3 channels, 384x384)
    dummy_input_image = torch.randn(1, 3, 384, 384)
    output_img, attn_img = model(dummy_input_image)
    print(f"Single Image Output shape: {output_img.shape}")
    print(f"Single Image Attention shape: {attn_img.shape}")
    
    # Dummy input video (1 batch, 5 frames, 3 channels, 384x384)
    dummy_input_video = torch.randn(1, 5, 3, 384, 384)
    output_vid, attn_vid = model(dummy_input_video)
    print(f"Video Output shape: {output_vid.shape}")
    print(f"Video Attention shape: {attn_vid.shape}")
