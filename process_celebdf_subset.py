import os
import shutil
from tqdm import tqdm
from utils.video_processing import extract_frames_from_video

def process_subset(input_dir, output_dir, max_videos=20, max_frames=5):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    if not os.path.exists(input_dir):
        print(f"Skipping {input_dir} - does not exist.")
        return
        
    all_videos = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    subset = all_videos[:max_videos]
    
    print(f"Processing {len(subset)} videos from {input_dir}...")
    
    total_extracted = 0
    for video_file in tqdm(subset, desc=f"Extracting {os.path.basename(input_dir)}"):
        video_path = os.path.join(input_dir, video_file)
        vid_name = os.path.splitext(video_file)[0]
        
        extracted_images = extract_frames_from_video(video_path, max_frames=max_frames, output_dir=None)
        
        for frame_idx, img in enumerate(extracted_images):
            save_path = os.path.join(output_dir, f"{vid_name}_frame_{frame_idx:04d}.jpg")
            img.save(save_path)
            total_extracted += 1
            
    print(f"Done! Extracted {total_extracted} images to {output_dir}")

def main():
    print("Starting Celeb-DF subset processing...")
    
    # Process Real Videos
    process_subset('Celeb-DF-v2/Celeb-real', 'dataset/benchmark_data/real', max_videos=20, max_frames=5)
    
    # Process Fake Videos
    process_subset('Celeb-DF-v2/Celeb-synthesis', 'dataset/benchmark_data/fake', max_videos=20, max_frames=5)
    
if __name__ == "__main__":
    main()
