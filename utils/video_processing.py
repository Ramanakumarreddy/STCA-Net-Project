import cv2
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Haar Cascades are the fastest CPU-compatible face detector and are accurate
# enough for frontal-face deepfake analysis. We load a single shared cascade.
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def get_face_cascade():
    """
    Load the OpenCV Haar Cascade frontal-face detector.

    Returns:
        cv2.CascadeClassifier: The loaded face detector.

    Raises:
        FileNotFoundError: If the Haar cascade XML file cannot be found
                           in the OpenCV installation data directory.
    """
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        logger.error("Failed to load Haar cascade classifier.")
        raise FileNotFoundError("Haar cascade XML file not found.")
    return cascade

def extract_face(image_array, cascade):
    """
    Detect the largest face in an image array and return it as a PIL Image.

    Detection uses the provided Haar Cascade classifier. The largest face
    (by bounding-box area) is selected, and a 20% margin is added on all
    sides to include forehead, chin, and cheek context. If no face is found,
    a square center crop of the image is returned as a fallback.

    Args:
        image_array (np.ndarray): RGB image array of shape (H, W, 3).
        cascade (cv2.CascadeClassifier): A loaded Haar Cascade classifier.

    Returns:
        PIL.Image.Image: The cropped face region (or center crop if no face found).
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
    Extract a sequence of sharp, face-cropped frames from a video file.

    This function implements a **smart frame selection** strategy to avoid
    blurry or redundant frames:
        1. The video is divided into ``max_frames`` equal-duration intervals.
        2. Within each interval, up to 5 candidate frames are sampled.
        3. The sharpest candidate (highest Variance of Laplacian score) is selected.
        4. Haar Cascade face detection is applied to the chosen frame; the largest
           detected face (+ 20% margin) is cropped, or a center crop is used as fallback.

    This matches the inference preprocessing in ``utils/prediction.py`` so that
    the model receives consistently face-centred inputs during both training and
    real-time video analysis.

    Args:
        video_path (str): Path to the input video file (supports .mp4, .avi, .mov, etc.).
        max_frames (int): Maximum number of frames to extract. Frames are distributed
                          evenly across the video duration. Default: 15.
        output_dir (str | None): If provided, each extracted face image is saved as a
                                 JPEG to this directory (created if absent). Default: None.

    Returns:
        list[PIL.Image.Image]: A list of face-cropped PIL Images in chronological order.
                               May be shorter than ``max_frames`` if the video is very
                               short or if frames could not be decoded.
                               Returns an empty list on any unrecoverable file error.
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
        
    for i in range(max_frames):
        start_frame = i * interval
        end_frame = min((i + 1) * interval, total_frames)
        
        best_frame = None
        max_sharpness = -1.0
        
        # Sample up to 'sample_limit' frames within this interval to find the sharpest one
        # Avoid processing every single frame for long videos to maintain performance
        sample_limit = min(5, end_frame - start_frame)
        if sample_limit < 1:
            continue
            
        step = max(1, (end_frame - start_frame) // sample_limit)
        
        for f in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to gray to check sharpness using Variance of Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_frame = frame
                
        if best_frame is not None:
            # We found the sharpest frame in this interval, now extract the face
            rgb_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            face_img = extract_face(rgb_frame, cascade)
            
            if face_img is not None:
                extracted_images.append(face_img)
                
                # Save to disk if output_dir is provided
                if output_dir:
                    save_path = os.path.join(output_dir, f"frame_{len(extracted_images):04d}.jpg")
                    face_img.save(save_path)
        
    cap.release()
    logger.info(f"Extracted {len(extracted_images)} smart-selected face frames from video.")
    return extracted_images
