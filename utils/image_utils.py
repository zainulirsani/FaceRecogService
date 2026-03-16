"""
utils/image_utils.py — Image Processing Helper
================================================
Semua operasi konversi & manipulasi gambar ada di sini
supaya service layer (encoder, matcher) tetap bersih.
"""

import base64
import numpy as np
import cv2
from typing import Tuple, Optional
from config import MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT


def decode_base64_image(b64_string: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Decode string base64 menjadi numpy array (format BGR untuk OpenCV).

    Laravel mengirim foto dari kamera HP sebagai:
        data:image/jpeg;base64,/9j/4AAQSkZJRg...
    atau langsung tanpa prefix:
        /9j/4AAQSkZJRg...

    Returns:
        (image_array, None)        → sukses
        (None, "error message")    → gagal
    """
    try:
        # Hapus data URL prefix jika ada (dari browser camera API)
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        # Hapus whitespace/newline yang kadang muncul
        b64_string = b64_string.strip()

        # Decode base64 → bytes
        img_bytes = base64.b64decode(b64_string)

        # Bytes → numpy array → decode sebagai gambar
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return None, "Failed to decode image bytes (invalid format)"

        # Resize jika terlalu besar (hemat memory di Render.com)
        image = resize_if_needed(image)

        return image, None

    except base64.binascii.Error:
        return None, "Invalid base64 string"
    except Exception as e:
        return None, str(e)


def resize_if_needed(image: np.ndarray) -> np.ndarray:
    """
    Resize gambar jika melebihi MAX_IMAGE_WIDTH/HEIGHT.
    Mempertahankan aspect ratio.
    """
    h, w = image.shape[:2]

    if w <= MAX_IMAGE_WIDTH and h <= MAX_IMAGE_HEIGHT:
        return image

    # Hitung scale factor
    scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR (OpenCV default) ke RGB (face_recognition default).
    face_recognition library menggunakan format RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert ke grayscale, dipakai untuk analisis texture anti-spoofing."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
