"""
Comprehensive End-to-End Evaluation Script.
Randomly samples videos from across the ENTIRE Celeb-DF dataset,
extracts fresh frames, and evaluates the trained STCA-Net model.
"""
import os
import random
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.stca_net import STCANet
from utils.video_processing import extract_frames_from_video

# ── Config ────────────────────────────────────────────────────
EVAL_REAL_SAMPLE  = 50     # Real videos to randomly sample
EVAL_FAKE_SAMPLE  = 50     # Fake videos to randomly sample
FRAMES_PER_VIDEO  = 5      # Frames extracted per video
EVAL_DIR          = "dataset/eval_comprehensive"
WEIGHTS_PATH      = "models/stca_net_weights.pt"
BATCH_SIZE        = 16
SEED              = 42
# ─────────────────────────────────────────────────────────────

random.seed(SEED)

def get_video_files(directory):
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid_exts)]

def extract_to_dir(video_list, out_dir, max_frames):
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for vp in video_list:
        vname = os.path.splitext(os.path.basename(vp))[0]
        try:
            frames = extract_frames_from_video(vp, max_frames=max_frames, output_dir=None)
            for i, img in enumerate(frames):
                img.save(os.path.join(out_dir, f"{vname}_f{i:04d}.jpg"))
                total += 1
        except Exception as e:
            print(f"  [warn] {os.path.basename(vp)}: {e}")
    return total

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.samples = []
        for cls, label in [("fake", 0), ("real", 1)]:
            d = os.path.join(root_dir, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(d, f), label))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def compute_metrics(model, loader, device):
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
    return tp, tn, fp, fn, total, acc, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # 1. Load model
    if not os.path.exists(WEIGHTS_PATH):
        print("ERROR: weights not found at", WEIGHTS_PATH)
        return
    model = STCANet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"Weights : {WEIGHTS_PATH}")

    # 2. Scan entire dataset
    print("\nScanning Celeb-DF-v2 ...")
    all_real = (get_video_files("Celeb-DF-v2/Celeb-real") +
                get_video_files("Celeb-DF-v2/YouTube-real"))
    all_fake =  get_video_files("Celeb-DF-v2/Celeb-synthesis")
    print(f"  Real videos available: {len(all_real)}")
    print(f"  Fake videos available: {len(all_fake)}")

    # 3. Random sample spread across the full dataset
    sampled_real = random.sample(all_real, min(EVAL_REAL_SAMPLE, len(all_real)))
    sampled_fake = random.sample(all_fake, min(EVAL_FAKE_SAMPLE, len(all_fake)))

    # 4. Clean output dir
    if os.path.exists(EVAL_DIR):
        shutil.rmtree(EVAL_DIR)

    print(f"\nExtracting {FRAMES_PER_VIDEO} frames from {len(sampled_real)} real videos ...")
    n_real = extract_to_dir(sampled_real, os.path.join(EVAL_DIR, "real"), FRAMES_PER_VIDEO)

    print(f"Extracting {FRAMES_PER_VIDEO} frames from {len(sampled_fake)} fake videos ...")
    n_fake = extract_to_dir(sampled_fake, os.path.join(EVAL_DIR, "fake"), FRAMES_PER_VIDEO)

    print(f"  -> {n_real} real  frames | {n_fake} fake frames  (total: {n_real+n_fake})")

    # 5. Build DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = EvalDataset(EVAL_DIR, transform)
    if len(ds) == 0:
        print("ERROR: No images were extracted. Check video paths.")
        return
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Evaluate
    print(f"\nRunning inference on {len(ds)} frames ...")
    tp, tn, fp, fn, total, acc, prec, rec, f1 = compute_metrics(model, loader, device)

    # 7. Report
    bar = "=" * 60
    print(f"\n{bar}")
    print("  COMPREHENSIVE END-TO-END EVALUATION RESULTS")
    print(bar)
    print(f"  Videos sampled  : {len(sampled_real)} real  +  {len(sampled_fake)} fake")
    print(f"  Frames evaluated: {total}  (real={n_real}, fake={n_fake})")
    print(f"{bar}")
    print(f"  True Positives  (Real->Real) : {tp}")
    print(f"  True Negatives  (Fake->Fake) : {tn}")
    print(f"  False Positives (Fake->Real) : {fp}")
    print(f"  False Negatives (Real->Fake) : {fn}")
    print(f"{bar}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(bar)

    # Verdict
    if acc >= 0.85:
        verdict = "EXCELLENT - Model is production ready."
    elif acc >= 0.75:
        verdict = "GOOD - More training on diverse data recommended."
    elif acc >= 0.60:
        verdict = "FAIR - Model needs significant additional training."
    else:
        verdict = "POOR - Model has not generalised well."
    print(f"  Verdict : {verdict}")
    print(bar)

    # 8. Cleanup extracted eval frames
    shutil.rmtree(EVAL_DIR)
    print("\nEval frames cleaned up.")

if __name__ == "__main__":
    main()
