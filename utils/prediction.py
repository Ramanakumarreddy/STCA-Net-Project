import os
import cv2
import numpy as np
import torch
import logging
from torchvision import transforms
from PIL import Image
from scipy.fft import dctn

logger = logging.getLogger(__name__)

# ── Preprocessing transform ────────────────────────────────────────────────────
# Both MobileNetV3 and ViT sub-networks expect ImageNet-normalized 384×384 input.
preprocess_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Haar Cascade (lazy-loaded singleton) ─────────────────────────────────────────
# Loading the XML file is expensive; we do it once and cache it.
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_face_cascade = None

def _get_face_cascade():
    """
    Lazily load and cache the OpenCV Haar Cascade face detector.

    Returns:
        cv2.CascadeClassifier: The loaded classifier, ready to call detectMultiScale().

    Raises:
        FileNotFoundError: If the Haar cascade XML file cannot be found in the
                           OpenCV data directory.
    """
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if _face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier.")
            raise FileNotFoundError("Haar cascade XML file not found.")
    return _face_cascade


def extract_face_from_image(pil_image):
    """
    Detect and crop the largest face from a PIL Image using Haar Cascades.

    A 20% margin is added around the detected bounding box to include
    chin, forehead, and cheek context. If no face is detected, a square
    center crop of the image is returned as a fallback so inference can
    still proceed.

    Args:
        pil_image (PIL.Image.Image): The input image in RGB mode.

    Returns:
        tuple:
            - cropped (PIL.Image.Image): The face region (or center crop).
            - face_found (bool): True if a face was detected, False for center crop.
    """
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    if len(faces) == 0:
        # No face found — center crop as fallback
        h, w = img_array.shape[:2]
        size = min(h, w)
        y1, x1 = (h - size) // 2, (w - size) // 2
        cropped = img_array[y1:y1+size, x1:x1+size]
        return Image.fromarray(cropped), False
    
    # Find the largest face by area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add 20% margin around the face for context
    margin_w, margin_h = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_array.shape[1], x + w + margin_w)
    y2 = min(img_array.shape[0], y + h + margin_h)
    
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped), True


def compute_frequency_score(pil_image):
    """
    Compute a frequency-domain anomaly score using 2D Discrete Cosine Transform (DCT).

    AI-generated images (GANs, diffusion models) share a distinctive spectral
    signature compared to real photographs:
        - **Lower high-frequency energy** — smoother, overly-clean textures.
        - **Concentrated low-frequency energy** — energy is over-centralised.
        - **Steeper spectral decay slope** — energy drops off faster from low to high frequencies.

    The scoring heuristic is calibrated empirically and clamps to [0.0, 1.0].

    Args:
        pil_image (PIL.Image.Image): Input image in any PIL mode (auto-converted to grayscale).

    Returns:
        float: Anomaly score in [0.0, 1.0].
               0.0 = no AI-generated frequency signature detected (looks natural/real).
               1.0 = strong AI-generated frequency signature.
               Returns 0.0 silently on any processing error.
    """
    try:
        # Convert to grayscale numpy array
        img_gray = np.array(pil_image.convert('L').resize((256, 256)), dtype=np.float64)
        
        # Apply 2D Discrete Cosine Transform
        dct_coeffs = dctn(img_gray, norm='ortho')
        
        # Compute power spectrum (log magnitude)
        power = np.abs(dct_coeffs)
        
        h, w = power.shape
        
        # Split into frequency bands
        # Low frequency: top-left quadrant
        low_freq = power[:h//4, :w//4]
        # Mid frequency: between low and high
        mid_freq_mask = np.zeros_like(power, dtype=bool)
        mid_freq_mask[h//4:h//2, :w//2] = True
        mid_freq_mask[:h//2, w//4:w//2] = True
        mid_freq = power[mid_freq_mask]
        # High frequency: bottom-right region
        high_freq = power[h//2:, w//2:]
        
        total_energy = np.sum(power) + 1e-10
        low_energy = np.sum(low_freq) / total_energy
        mid_energy = np.sum(mid_freq) / total_energy
        high_energy = np.sum(high_freq) / total_energy
        
        # AI-generated images typically have:
        # 1. Lower high-frequency energy ratio (smoother textures)
        # 2. More concentrated energy in low frequencies
        # 3. Smoother spectral decay
        
        # Compute spectral decay rate (row-wise average of DCT magnitudes)
        row_energies = np.mean(power, axis=1)
        if len(row_energies) > 1:
            # Fit log decay — steeper = more AI-like
            log_energies = np.log(row_energies + 1e-10)
            x = np.arange(len(log_energies))
            # Simple linear fit
            slope = np.polyfit(x, log_energies, 1)[0]
        else:
            slope = 0
        
        # Scoring heuristic (calibrated empirically):
        # High-freq ratio < 0.08 is suspicious (AI-like)
        # Spectral slope < -0.06 is suspicious
        score = 0.0
        
        if high_energy < 0.05:
            score += 0.4
        elif high_energy < 0.08:
            score += 0.2
            
        # Removed the penalty for high energy, as modern diffusion models (like Midjourney/Gemini) 
        # can produce exceptionally sharp images, causing false negatives.
        
        if slope < -0.08:
            score += 0.3
        elif slope < -0.06:
            score += 0.15
        
        if low_energy > 0.85:
            score += 0.3
        elif low_energy > 0.75:
            score += 0.15
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
        
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        return 0.0  # Default to no signal


def detect_non_photographic(pil_image):
    """
    Detect whether an image is non-photographic (anime, cartoon, illustration, etc.).

    Uses four image statistics to identify non-photographic content:
        1. **Saturation** — cartoons/anime have high mean saturation and low variance.
        2. **Unique colour count (quantized)** — fewer distinct colours suggest flat artwork.
        3. **Texture variance** (Laplacian) — artwork has lower overall texture complexity.
        4. **Edge density** (Canny) — sharp outlines with flat fill areas are cartoon-like.

    The combined score threshold is 0.5. This check is important to avoid
    false-positive FAKE predictions on clearly non-photographic inputs where
    the deepfake detection model is out-of-distribution.

    Args:
        pil_image (PIL.Image.Image): Input image in any PIL mode.

    Returns:
        tuple:
            - is_non_photo (bool): True if the image appears non-photographic.
            - confidence (float): Rounded composite score in [0.0, 1.0].
                                  Returns (False, 0.0) on any processing error.
    """
    try:
        img_array = np.array(pil_image.convert('RGB').resize((256, 256)))
        
        # 1. Color histogram analysis — cartoons/anime have fewer unique colors
        #    and more saturated, uniform color blocks
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Mean and std of saturation
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # 2. Edge density — cartoons have sharp, clean edges with flat areas
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # 3. Unique color count (quantized)
        quantized = (img_array // 32) * 32  # Reduce to ~8 levels per channel
        flat = quantized.reshape(-1, 3)
        unique_colors = len(np.unique(flat, axis=0))
        
        # 4. Texture analysis — real photos have more texture variation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # Scoring: anime/cartoons typically have:
        # - High saturation mean (>80) with low std
        # - Moderate edge density with very flat inter-edge regions
        # - Fewer unique quantized colors (<200)
        # - Lower texture variance
        
        anime_score = 0.0
        
        if sat_mean > 90 and sat_std < 50:
            anime_score += 0.3
        
        if unique_colors < 150:
            anime_score += 0.3
        elif unique_colors < 250:
            anime_score += 0.15
        
        if texture_var < 500:
            anime_score += 0.2
        
        if edge_density > 0.05 and texture_var < 800:
            anime_score += 0.2
        
        is_non_photo = anime_score >= 0.5
        
        return is_non_photo, round(anime_score, 2)
        
    except Exception as e:
        logger.warning(f"Non-photographic detection failed: {e}")
        return False, 0.0


def check_ai_signatures(image_path, pil_image=None):
    """
    Perform a fast heuristic scan for explicit AI-generation signatures in a file's
    name and EXIF/metadata, before invoking any neural network inference.

    Checks:
        1. **Filename keywords**: Scans the basename for common AI tool names
           (e.g., ``gemini_generated``, ``dalle``, ``midjourney``, ``stable_diffusion``).
        2. **Image metadata/EXIF**: Inspects PIL's ``image.info`` dict as a string
           for known generator tags from Google Gemini, DALL-E, Midjourney, and
           Stable Diffusion.

    Args:
        image_path (str): Absolute or relative path to the image file on disk.
        pil_image (PIL.Image.Image, optional): The opened PIL image (used to inspect its .info dict).

    Returns:
        tuple:
            - sig_score (float): 1.0 if an AI signature was found, 0.0 otherwise.
            - sig_reason (str): Human-readable description of the matched signature,
                                or an empty string if no signature was found.
    """
    try:
        # 1. Check filename
        if image_path:
            basename = os.path.basename(image_path).lower()
            ai_keywords = ['gemini_generated', 'dall_e', 'dalle', 'midjourney', 'stable_diffusion', 'ai_generated', 'journey']
            if any(k in basename for k in ai_keywords):
                return 1.0, "AI signature found in filename."
            
        # 2. Check metadata / EXIF
        if pil_image:
            info_str = str(pil_image.info).lower()
            if 'google' in info_str and 'gemini' in info_str:
                return 1.0, "AI signature found in metadata (Gemini)."
            elif 'dall-e' in info_str or 'midjourney' in info_str or 'stable diffusion' in info_str:
                return 1.0, "AI generation software found in metadata."
            
    except Exception as e:
        logger.warning(f"Signature check failed: {e}")
        pass
        
    return 0.0, ""


def predict_image(model, image_path, device='cpu'):
    """
    Run the full STCA-Net deepfake detection pipeline on a single image.

    Pipeline stages (in order):
        1. AI signature check — filename + EXIF metadata scan.
        2. Non-photographic detection — saturation / edge / texture analysis.
        3. Face extraction — Haar Cascade crop (falls back to center crop).
        4. DCT frequency analysis — band-energy AI-likeness score.
        5. STCA-Net forward pass — on the face-cropped, 384×384 image.
        6. Score fusion — NN (70%) + frequency (30%), with signature override.

    If an explicit AI generation signature is found (step 1), the score is
    overridden to 95% FAKE without running the neural network.

    Args:
        model (STCANet): A loaded (and optionally `.eval()`-ed) STCA-Net model.
        image_path (str): Path to the image file to analyse.
        device (str | torch.device): PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.

    Returns:
        dict: A result dictionary with the following keys:
            - ``prediction`` (str): ``'REAL'`` or ``'FAKE'``.
            - ``confidence`` (float): Percentage confidence of the predicted class.
            - ``fake_probability`` (float): Combined FAKE probability (%).
            - ``real_probability`` (float): Combined REAL probability (%).
            - ``nn_fake_probability`` (float): Neural network FAKE probability only (%).
            - ``nn_real_probability`` (float): Neural network REAL probability only (%).
            - ``attention_map`` (np.ndarray): Cross-attention weights (B, T, 1, 144).
            - ``face_detected`` (bool): Whether a face was found by Haar Cascade.
            - ``frequency_score`` (float): DCT AI-likeness score 0–100 (higher = more AI-like).
            - ``signature_found`` (bool): True if an AI metadata/filename signature was detected.
            - ``signature_reason`` (str): Description of the signature match (if any).
            - ``is_non_photographic`` (bool): True if classified as cartoon/anime/illustration.
            - ``warning`` (str, optional): Present only when ``is_non_photographic`` is True.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        Exception: If any processing step fails (wraps the original error message).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Check for obvious AI signatures (Watermark/Metadata/Filename)
        sig_score, sig_reason = check_ai_signatures(image_path, img)
        
        # Check for non-photographic content (anime, cartoon, etc.)
        is_non_photo, non_photo_score = detect_non_photographic(img)
        
        # Extract face from image (consistent with video pipeline)
        face_img, face_found = extract_face_from_image(img)
        
        # Compute frequency-domain analysis score
        freq_score = compute_frequency_score(img)
        
        # Neural network prediction on face-cropped image
        input_tensor = preprocess_transform(face_img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output, attn_weights = model(input_tensor)
            
            # Assuming output is shape (1, 2)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Class 0: Fake, Class 1: Real
            nn_fake_prob = probabilities[0].item()
            nn_real_prob = probabilities[1].item()
        
        # ============================================================
        # COMBINED SCORING: Blend neural network + frequency analysis + signatures
        # ============================================================
        
        if sig_score == 1.0:
            # If an explicit AI signature is found, override the prediction
            combined_fake_prob = 0.95
            combined_real_prob = 0.05
        else:
            # Neural network weight: 0.7, Frequency analysis weight: 0.3
            nn_weight = 0.80
            freq_weight = 0.20
            
            # freq_score is 0=real, 1=AI-generated → treat as fake probability
            combined_fake_prob = (nn_fake_prob * nn_weight) + (freq_score * freq_weight)
            combined_real_prob = (nn_real_prob * nn_weight) + ((1 - freq_score) * freq_weight)
            
            # Normalize
            total = combined_fake_prob + combined_real_prob
            combined_fake_prob /= total
            combined_real_prob /= total
            
        fake_prob = combined_fake_prob * 100
        real_prob = combined_real_prob * 100
        
        prediction = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = real_prob if prediction == "REAL" else fake_prob
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_prob, 2),
            'real_probability': round(real_prob, 2),
            'nn_fake_probability': round(nn_fake_prob * 100, 2),
            'nn_real_probability': round(nn_real_prob * 100, 2),
            'attention_map': attn_weights.cpu().numpy(),
            'face_detected': face_found,
            'frequency_score': round(freq_score * 100, 2),
            'signature_found': sig_score == 1.0,
            'signature_reason': sig_reason
        }
        
        # Add non-photographic warning if detected
        if is_non_photo:
            result['warning'] = (
                "This appears to be non-photographic content (anime, cartoon, or illustration). "
                "Deepfake detection is designed for real photographs and may not be reliable for this type of image."
            )
            result['is_non_photographic'] = True
        else:
            result['is_non_photographic'] = False
        
        logger.info(
            f"Image prediction: {prediction} ({confidence:.1f}%) | "
            f"NN: fake={nn_fake_prob:.3f} real={nn_real_prob:.3f} | "
            f"Freq: {freq_score:.3f} | Sig: {sig_score} | Face: {face_found} | NonPhoto: {is_non_photo}"
        )
        
        return result
        
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")


def predict_video_frames(model, frames, device='cpu', video_path=None):
    """
    Run the full STCA-Net deepfake detection pipeline on a pre-extracted list of frames.

    Processes the frame list in two ways:
        1. **Sequence inference**: All frames are stacked into a single
           ``(1, T, C, H, W)`` tensor and passed through STCA-Net's temporal
           encoder for a coherent multi-frame decision.
        2. **Per-frame inference**: Each frame is also evaluated individually
           to produce ``per_frame_scores`` for timeline visualisation in the UI.

    DCT frequency analysis is run on every frame independently; the per-frame
    scores are averaged and blended with the NN score (70% NN / 30% freq).

    Args:
        model (STCANet): A loaded STCA-Net model. Should be in eval mode.
        frames (list[PIL.Image.Image]): List of face-cropped PIL Images, one per
                                        sampled video frame.
        device (str | torch.device): PyTorch device, e.g. ``'cpu'`` or ``'cuda'``.
        video_path (str, optional): Path to the video file for signature check.

    Returns:
        dict: A result dictionary with the following keys:
            - ``prediction`` (str): ``'REAL'`` or ``'FAKE'``.
            - ``confidence`` (float): Percentage confidence of the predicted class.
            - ``frames_analyzed`` (int): Number of frames processed.
            - ``fake_probability`` (float): Combined FAKE probability (%).
            - ``real_probability`` (float): Combined REAL probability (%).
            - ``nn_fake_probability`` (float): Sequence-level NN FAKE probability (%).
            - ``nn_real_probability`` (float): Sequence-level NN REAL probability (%).
            - ``frequency_score`` (float): Average DCT AI-likeness score 0–100.
            - ``per_frame_scores`` (list[float]): Per-frame FAKE probabilities (%) from individual NN passes.
            - ``per_frame_freq_scores`` (list[float]): Per-frame DCT AI-likeness scores (%).

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("No frames provided for prediction.")
        
    sig_score = 0.0
    sig_reason = ""
    if video_path:
        sig_score, sig_reason = check_ai_signatures(video_path)
        
    freq_scores = []
    tensor_frames = []
    
    for frame in frames:
        # Compute frequency score for each frame individually
        freq_scores.append(compute_frequency_score(frame))
        # Transform for neural network
        tensor_frames.append(preprocess_transform(frame))
        
    avg_freq = sum(freq_scores) / len(freq_scores)
    
    # Stack frames into a single sequence batch (1, T, C, H, W)
    input_sequence = torch.stack(tensor_frames).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # Pass the entire sequence to the spatio-temporal model
        output, _ = model(input_sequence)
        
        # output is shape (1, 2)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        avg_nn_fake = probabilities[0].item()
        avg_nn_real = probabilities[1].item()
        
        # Per-frame analysis: run each frame individually for timeline chart
        per_frame_fake_probs = []
        for tensor_frame in tensor_frames:
            single_input = tensor_frame.unsqueeze(0).to(device)  # (1, C, H, W)
            single_output, _ = model(single_input)
            single_probs = torch.nn.functional.softmax(single_output[0], dim=0)
            per_frame_fake_probs.append(round(single_probs[0].item() * 100, 2))
    
    # Combine with frequency analysis (same weighting as image)
    if sig_score == 1.0:
        combined_fake = 0.95
        combined_real = 0.05
    else:
        nn_weight = 0.70
        freq_weight = 0.30
        
        combined_fake = (avg_nn_fake * nn_weight) + (avg_freq * freq_weight)
        combined_real = (avg_nn_real * nn_weight) + ((1 - avg_freq) * freq_weight)
        
        total = combined_fake + combined_real
        combined_fake /= total
        combined_real /= total
    
    avg_fake_prob = combined_fake * 100
    avg_real_prob = combined_real * 100
    
    prediction = "REAL" if avg_real_prob > avg_fake_prob else "FAKE"
    confidence = avg_real_prob if prediction == "REAL" else avg_fake_prob
    
    return {
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'frames_analyzed': len(frames),
        'fake_probability': round(avg_fake_prob, 2),
        'real_probability': round(avg_real_prob, 2),
        'nn_fake_probability': round(avg_nn_fake * 100, 2),
        'nn_real_probability': round(avg_nn_real * 100, 2),
        'frequency_score': round(avg_freq * 100, 2),
        'per_frame_scores': per_frame_fake_probs,
        'per_frame_freq_scores': [round(s * 100, 2) for s in freq_scores],
        'signature_found': sig_score == 1.0,
        'signature_reason': sig_reason
    }
