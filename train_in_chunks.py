"""
train_in_chunks.py  —  Progressive chunk-based trainer with Replay Buffer.

How it works:
  1. Split the full dataset into 20-video chunks.
  2. For each chunk:
       a) Extract sharp frames from the current 20 real + 20 fake videos.
       b) Copy a random sample from the REPLAY BUFFER into the chunk folder
          so the model sees old AND new data in every training run.
       c) Train STCA-Net for 5 epochs (resume from saved weights).
       d) Update the replay buffer: randomly pick frames from this chunk
          to remember for next time, respecting a max buffer size.
       e) Wipe the chunk_data folder to reclaim disk space.
"""

import os
import random
import shutil
import subprocess
from itertools import cycle

from utils.video_processing import extract_frames_from_video

# ── Tunable constants ──────────────────────────────────────────
CHUNK_SIZE        = 20      # Videos per class per chunk
FRAMES_PER_VIDEO  = 5       # Sharp frames to extract per video
EPOCHS_PER_CHUNK  = 5       # Epochs trained on each chunk (was 3)

REPLAY_BUFFER_CAP = 200     # Max images per class kept in buffer
REPLAY_MIX_COUNT  = 40      # Images per class injected from buffer each chunk

PYTHON_EXE = os.path.join("venv", "Scripts", "python.exe") \
             if os.path.exists(os.path.join("venv", "Scripts", "python.exe")) \
             else "python"

CHUNK_DIR   = os.path.join("dataset", "chunk_data")
REPLAY_DIR  = os.path.join("dataset", "replay_buffer")
WEIGHTS     = os.path.join("models", "stca_net_weights.pt")
# ─────────────────────────────────────────────────────────────


def get_video_files(directory):
    valid = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid)]


def empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extract_videos_to_dir(video_list, out_dir, max_frames=5):
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


def inject_replay(buf_class_dir, dst_class_dir, n):
    """Copy up to n random frames from the replay buffer into dst_class_dir."""
    if not os.path.isdir(buf_class_dir):
        return 0
    imgs = [f for f in os.listdir(buf_class_dir) if f.lower().endswith('.jpg')]
    if not imgs:
        return 0
    chosen = random.sample(imgs, min(n, len(imgs)))
    for fname in chosen:
        src = os.path.join(buf_class_dir, fname)
        # prefix with "replay_" to avoid name collisions
        shutil.copy2(src, os.path.join(dst_class_dir, "replay_" + fname))
    return len(chosen)


def update_replay_buffer(chunk_class_dir, buf_class_dir, cap):
    """
    Sample fresh frames from this chunk into the replay buffer.
    Keeps the buffer under `cap` images (evicts oldest randomly if over).
    """
    os.makedirs(buf_class_dir, exist_ok=True)
    src_imgs = [f for f in os.listdir(chunk_class_dir)
                if f.lower().endswith('.jpg') and not f.startswith("replay_")]
    if not src_imgs:
        return

    # How many new frames to add: fill up to cap
    buf_imgs = [f for f in os.listdir(buf_class_dir) if f.lower().endswith('.jpg')]
    slots_free = cap - len(buf_imgs)

    if slots_free <= 0:
        # Buffer is full: evict random frames to make room for new ones
        to_evict = random.sample(buf_imgs, min(len(src_imgs), len(buf_imgs)))
        for f in to_evict:
            os.remove(os.path.join(buf_class_dir, f))

    to_add = random.sample(src_imgs, min(len(src_imgs), max(1, cap // 10)))
    for fname in to_add:
        shutil.copy2(os.path.join(chunk_class_dir, fname),
                     os.path.join(buf_class_dir, f"chunk_repr_{fname}"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chunk-based trainer with Replay Buffer")
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Cap total chunks (useful for testing)')
    parser.add_argument('--start-chunk', type=int, default=0,
                        help='Resume from this chunk index (0-based)')
    args = parser.parse_args()

    # ── Scan full dataset ─────────────────────────────────────
    print("Scanning Celeb-DF-v2 ...")
    real_videos = (get_video_files(os.path.join("Celeb-DF-v2", "Celeb-real")) +
                   get_video_files(os.path.join("Celeb-DF-v2", "YouTube-real")))
    fake_videos  =  get_video_files(os.path.join("Celeb-DF-v2", "Celeb-synthesis"))

    if not real_videos or not fake_videos:
        print("ERROR: Celeb-DF-v2 dataset not found.")
        return

    num_chunks = max(1, len(fake_videos) // CHUNK_SIZE)
    if args.max_chunks:
        num_chunks = min(num_chunks, args.max_chunks)

    print(f"Real videos : {len(real_videos)}")
    print(f"Fake videos : {len(fake_videos)}")
    print(f"Chunks      : {num_chunks}  (size={CHUNK_SIZE}, epochs/chunk={EPOCHS_PER_CHUNK})")
    print(f"Replay buf  : cap={REPLAY_BUFFER_CAP}/class, inject={REPLAY_MIX_COUNT}/class per chunk")
    print(f"Starting at chunk {args.start_chunk + 1}")

    real_cycle = cycle(real_videos)
    # fast-forward cycle if resuming mid-way
    for _ in range(args.start_chunk * CHUNK_SIZE):
        next(real_cycle)

    # ── Chunk loop ────────────────────────────────────────────
    for chunk_idx in range(args.start_chunk, num_chunks):
        print(f"\n{'='*55}")
        print(f"  CHUNK {chunk_idx + 1}/{num_chunks}")
        print(f"{'='*55}")

        real_out = os.path.join(CHUNK_DIR, "real")
        fake_out = os.path.join(CHUNK_DIR, "fake")

        # Step 1 — fresh extraction
        empty_dir(real_out)
        empty_dir(fake_out)

        current_fake = fake_videos[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
        current_real = [next(real_cycle) for _ in range(CHUNK_SIZE)]

        print(f"[1/4] Extracting frames ...")
        n_real = extract_videos_to_dir(current_real, real_out, FRAMES_PER_VIDEO)
        n_fake = extract_videos_to_dir(current_fake, fake_out, FRAMES_PER_VIDEO)
        print(f"      Real: {n_real} frames | Fake: {n_fake} frames")

        # Step 2 — inject replay buffer
        r_buf_real = os.path.join(REPLAY_DIR, "real")
        r_buf_fake = os.path.join(REPLAY_DIR, "fake")
        ri = inject_replay(r_buf_real, real_out, REPLAY_MIX_COUNT)
        fi = inject_replay(r_buf_fake, fake_out, REPLAY_MIX_COUNT)
        print(f"[2/4] Replay buffer injected: +{ri} real | +{fi} fake  "
              f"(total real={n_real+ri}, fake={n_fake+fi})")

        # Step 3 — train
        print(f"[3/4] Training for {EPOCHS_PER_CHUNK} epochs ...")
        cmd = [
            PYTHON_EXE, "train_stca_net.py",
            "--dataset",    CHUNK_DIR,
            "--epochs",     str(EPOCHS_PER_CHUNK),
            "--batch-size", "16",
            "--samples",    "0",           # Use ALL images in the chunk dir
        ]
        if os.path.exists(WEIGHTS):
            cmd += ["--resume-weights", WEIGHTS]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] Training failed on chunk {chunk_idx + 1}. Aborting.")
            break

        # Step 4 — update replay buffer
        print(f"[4/4] Updating replay buffer ...")
        update_replay_buffer(real_out, r_buf_real, REPLAY_BUFFER_CAP)
        update_replay_buffer(fake_out, r_buf_fake, REPLAY_BUFFER_CAP)

        buf_real_sz = len(os.listdir(r_buf_real)) if os.path.isdir(r_buf_real) else 0
        buf_fake_sz = len(os.listdir(r_buf_fake)) if os.path.isdir(r_buf_fake) else 0
        print(f"      Buffer size: real={buf_real_sz} | fake={buf_fake_sz}")

        # Wipe chunk data for disk space
        shutil.rmtree(CHUNK_DIR)
        print(f"[OK] Chunk {chunk_idx + 1} complete.")

    print("\n[DONE] All chunks processed. Training pipeline finished.")


if __name__ == "__main__":
    main()
