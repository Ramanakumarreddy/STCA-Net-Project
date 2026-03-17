import cv2
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Use Haar Cascades (fastest for CPU, good enough for most front-facing videos)
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def get_face_cascade():
    """Load the Haar cascade classifier."""
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        logger.error("Failed to load Haar cascade classifier.")
        raise FileNotFoundError("Haar cascade XML file not found.")
    return cascade

def extract_face(image_array, cascade):
    """
    Detects the largest face in an image array and returns the cropped face as a PIL Image.
    If no face is found, returns the center crop of the image.
    """
    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    if len(faces) == 0:
        # No face found, fallback to center crop
        h, w = image_array.shape[:2]
        size = min(h, w)
        y1, x1 = (h - size) // 2, (w - size) // 2
        cropped = image_array[y1:y1+size, x1:x1+size]
        return Image.fromarray(cropped)
        
    # Find the largest face by area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add a slight margin (20%) around the face to capture context
    margin_w, margin_h = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image_array.shape[1], x + w + margin_w)
    y2 = min(image_array.shape[0], y + h + margin_h)
    
    cropped = image_array[y1:y2, x1:x2]
    return Image.fromarray(cropped)

def extract_frames_from_video(video_path, max_frames=15, output_dir=None):
    """
    Extracts a sequence of face-cropped frames from a video file.
    Args:
        video_path: Path to the video file
        max_frames: Number of evenly spaced frames to extract
        output_dir: If provided, saves the extracted frames to disk
    Returns:
        List of PIL Images containing the cropped faces
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        logger.error("Invalid video format or unreadable frames.")
        cap.release()
        return []
        
    # Calculate interval to get evenly spaced frames
    interval = max(1, total_frames // max_frames)
    
    extracted_images = []
    cascade = get_face_cascade()
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break
            
        if frame_count % interval == 0:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract face
            face_img = extract_face(rgb_frame, cascade)
            extracted_images.append(face_img)
            
            # Save to disk if output_dir is provided (mostly for debugging/UI)
            if output_dir:
                save_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                face_img.save(save_path)
                
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    logger.info(f"Extracted {len(extracted_images)} face frames from video.")
    return extracted_images
