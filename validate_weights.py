"""Quick validation script to verify the trained weights load correctly."""
import torch
import os
from models.stca_net import STCANet

print("=" * 60)
print("STCA-Net Weight Validation")
print("=" * 60)

MODEL_PATH = "models/stca_net_weights.pt"

# Check file exists and size
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Weight file: {MODEL_PATH}")
    print(f"File size: {size_mb:.2f} MB")
else:
    print(f"ERROR: Weight file not found at {MODEL_PATH}")
    exit(1)

# Load model architecture
model = STCANet()
print(f"\nModel architecture: STCANet")
print(f"Total trainable parameters: {model.get_parameter_count():,}")

# Load weights
state_dict = torch.load(MODEL_PATH, map_location='cpu')
print(f"\nCheckpoint contains {len(state_dict)} state dict entries")
print(f"Sample keys: {list(state_dict.keys())[:5]}")

# Check for any key mismatches
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())

missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

if missing:
    print(f"\nWARNING: {len(missing)} keys missing from checkpoint:")
    for k in sorted(missing)[:10]:
        print(f"  - {k}")
if unexpected:
    print(f"\nWARNING: {len(unexpected)} unexpected keys in checkpoint:")
    for k in sorted(unexpected)[:10]:
        print(f"  - {k}")

if not missing and not unexpected:
    print("\nAll keys match perfectly!")

# Load into model
model.load_state_dict(state_dict)
print("\nWeights loaded successfully into model!")

# Run inference test
model.eval()
print("\n--- Inference Test ---")
dummy = torch.randn(1, 3, 384, 384)
with torch.no_grad():
    output, attn = model(dummy)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Fake probability: {probs[0].item():.4f}")
    print(f"Real probability: {probs[1].item():.4f}")

# Load training history
import json
history_path = "models/training_history.json"
if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
    epochs = len(history.get('train_loss', []))
    best_val_acc = max(history.get('val_acc', [0]))
    final_train_acc = history['train_acc'][-1] if history.get('train_acc') else 0
    final_val_acc = history['val_acc'][-1] if history.get('val_acc') else 0
    final_train_loss = history['train_loss'][-1] if history.get('train_loss') else 0
    final_val_loss = history['val_loss'][-1] if history.get('val_loss') else 0
    
    print(f"\n--- Training History ---")
    print(f"Total epochs trained: {epochs}")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final train accuracy: {final_train_acc*100:.2f}%")
    print(f"Final val loss: {final_val_loss:.4f}")
    print(f"Final val accuracy: {final_val_acc*100:.2f}%")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE - All checks passed!")
print("=" * 60)
