"""
Validation script: loads the trained STCA-Net weights and evaluates
on the current chunk_data and benchmark_data to produce accuracy,
precision, recall, F1, and a per-class breakdown.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.stca_net import STCANet

# ── dataset helper ──────────────────────────────────────────────
class SimpleEvalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.samples = []
        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")
        self._scan(fake_dir, label=0)
        self._scan(real_dir, label=1)

    def _scan(self, d, label):
        if not os.path.isdir(d):
            return
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(d, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

# ── evaluation function ────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out, _ = model(imgs)
            preds = out.argmax(dim=1)
            for p, l in zip(preds, labels):
                if   p == 1 and l == 1: tp += 1
                elif p == 0 and l == 0: tn += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "total": total}

# ── main ───────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = os.path.join("model", "stca_net_weights.pt")
    if not os.path.exists(weights_path):
        print("ERROR: No weights found at", weights_path)
        return

    # Load model
    model = STCANet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Evaluate on every available data source
    data_dirs = [
        ("dataset/chunk_data",      "Chunk Data (last chunk)"),
        ("dataset/benchmark_data",  "Benchmark Data (initial Celeb-DF subset)"),
    ]

    print("\n" + "=" * 60)
    print("STCA-Net Model Validation Report")
    print("=" * 60)

    for data_dir, label in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"\n--- {label}: SKIPPED (directory missing) ---")
            continue
        ds = SimpleEvalDataset(data_dir, val_transform)
        if len(ds) == 0:
            print(f"\n--- {label}: SKIPPED (no images) ---")
            continue
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        r = evaluate(model, loader, device)
        print(f"\n--- {label} ({r['total']} images) ---")
        print(f"  TP (Real->Real): {r['tp']}  |  TN (Fake->Fake): {r['tn']}")
        print(f"  FP (Fake->Real): {r['fp']}  |  FN (Real->Fake): {r['fn']}")
        print(f"  Accuracy : {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall   : {r['recall']:.4f}")
        print(f"  F1 Score : {r['f1']:.4f}")

    # Also show training history summary
    hist_path = os.path.join("model", "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        epochs = len(hist.get("train_loss", []))
        best_val = max(hist.get("val_acc", [0]))
        last_train = hist["train_acc"][-1] if hist.get("train_acc") else "N/A"
        print(f"\n--- Training History (last chunk) ---")
        print(f"  Epochs recorded : {epochs}")
        print(f"  Last train acc  : {last_train}")
        print(f"  Best val acc    : {best_val}")

    print("\n" + "=" * 60)
    print("Validation complete.")

if __name__ == "__main__":
    main()
