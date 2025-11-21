import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple


def findCircle(image: np.ndarray) -> list:
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=image.shape[0] // 2,
        param1=50,
        param2=30,
        minRadius=int(image.shape[0] * 0.2),
        maxRadius=int(image.shape[0] * 0.6)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return [int(x), int(y), int(r)]
    else:
        # Fallback to center
        h, w = image.shape[:2]
        return [w // 2, h // 2, min(h, w) // 2]


def cropImage(img: Image.Image, coordinates: list) -> Image.Image:
    x, y, r = coordinates
    padding = int(r * 0.05)
    r_padded = r + padding
    
    left = max(0, x - r_padded)
    top = max(0, y - r_padded)
    right = min(img.width, x + r_padded)
    bottom = min(img.height, y + r_padded)
    
    return img.crop((left, top, right, bottom))


def compute_frequency_features(gray_image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:

    if mask is not None:
        gray_image = gray_image * (mask > 0)
    
    # 2D FFT
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    dc_mask = distance > 5
    
    avg_frequency = np.sum(magnitude_spectrum * dc_mask * distance) / \
                   (np.sum(magnitude_spectrum * dc_mask) + 1e-10)
    
    high_freq_mask = distance > (min(rows, cols) * 0.3)
    high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask) / \
                      (np.sum(magnitude_spectrum) + 1e-10)
    
    return {
        'avg_frequency': float(avg_frequency),
        'high_freq_energy': float(high_freq_energy),
        'spectral_mean': float(np.mean(magnitude_spectrum[dc_mask])),
        'spectral_std': float(np.std(magnitude_spectrum[dc_mask]))
    }


def compute_intensity_features(img: Image.Image, gray: np.ndarray) -> Dict[str, float]:
    img_np = np.array(img)
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    features = {
        'mean_intensity': float(np.mean(gray)),
        'std_intensity': float(np.std(gray)),
        'mean_r': float(np.mean(img_np[:, :, 0])),
        'mean_g': float(np.mean(img_np[:, :, 1])),
        'mean_b': float(np.mean(img_np[:, :, 2])),
        'mean_hue': float(np.mean(hsv[:, :, 0])),
        'mean_saturation': float(np.mean(hsv[:, :, 1])),
        'mean_value': float(np.mean(hsv[:, :, 2])),
        'contrast': float(np.std(gray) / (np.mean(gray) + 1e-10)),
        'intensity_range': float(np.max(gray) - np.min(gray))
    }
    
    return features


def extract_features(img_pil: Image.Image) -> Dict[str, float]:

    img_np = np.array(img_pil)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    coordinates = findCircle(gray)
    
    # circular mask
    h, w = gray.shape
    y, x = np.ogrid[:h, :w]
    cx, cy, r = coordinates
    mask = ((x - cx)**2 + (y - cy)**2 <= r**2).astype(np.uint8) * 255
    
    # features
    freq_features = compute_frequency_features(gray, mask)
    intensity_features = compute_intensity_features(img_pil, gray)
    
    # (4 + 10 = 14)???
    all_features = {**freq_features, **intensity_features}
    
    return all_features


def preprocess_image(img_path: str) -> Tuple[Image.Image, Dict[str, float]]:
    img = Image.open(img_path).convert('RGB')
    
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    coordinates = findCircle(gray)
    img_cropped = cropImage(img, coordinates)
    
    features = extract_features(img_cropped)
    
    return img_cropped, features