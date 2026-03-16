"""
services/face_encoder.py — Face Encoding Generator
====================================================
Bertanggung jawab mendeteksi wajah dalam gambar dan
menghasilkan 128-dimensi face encoding vector menggunakan dlib.
"""

import face_recognition
import numpy as np
from typing import Tuple, Optional, List
from utils.image_utils import bgr_to_rgb


class FaceEncoder:
    """
    Menggunakan face_recognition library (wrapper dlib HOG + ResNet model).
    Model berjalan di CPU → cocok untuk Render.com free tier.
    """

    def encode(self, image_bgr: np.ndarray) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Deteksi wajah dalam gambar dan hasilkan face encoding.

        Args:
            image_bgr: numpy array format BGR (dari OpenCV)

        Returns:
            (encoding_list, None)       → sukses, encoding = list 128 float
            (None, "error message")     → gagal
        """
        # face_recognition butuh format RGB
        image_rgb = bgr_to_rgb(image_bgr)

        # Deteksi lokasi semua wajah dalam gambar
        # model="hog" → lebih cepat, cocok untuk CPU
        # model="cnn" → lebih akurat tapi butuh GPU
        face_locations = face_recognition.face_locations(image_rgb, model="hog")

        # Validasi: harus ada tepat 1 wajah
        if len(face_locations) == 0:
            return None, "No face detected in the image"

        if len(face_locations) > 1:
            return None, f"Multiple faces detected ({len(face_locations)}). Please use photo with single face only"

        # Generate encoding untuk wajah yang ditemukan
        encodings = face_recognition.face_encodings(image_rgb, face_locations)

        if not encodings:
            return None, "Failed to generate face encoding"

        # Konversi numpy array ke list biasa (agar bisa di-JSON-serialize)
        encoding_list = encodings[0].tolist()

        return encoding_list, None

    def encoding_to_array(self, encoding_list: List[float]) -> np.ndarray:
        """
        Konversi list (dari JSON) kembali ke numpy array.
        Dipakai FaceMatcher saat loading dari storage.
        """
        return np.array(encoding_list)
