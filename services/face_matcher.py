"""
services/face_matcher.py — Face Encoding Storage & Comparison
==============================================================
Bertanggung jawab menyimpan, memuat, dan membandingkan
face encoding antara foto enroll dan foto absen.
"""

import json
import os
import numpy as np
import face_recognition
from typing import Tuple, Optional, List, Dict, Any
from config import ENCODINGS_FILE, STORAGE_DIR, FACE_DISTANCE_THRESHOLD
from services.face_encoder import FaceEncoder


class FaceMatcher:

    def __init__(self):
        self.encoder  = FaceEncoder()
        self._ensure_storage()

    # ── Storage Helpers ───────────────────────────────────────────────────────

    def _ensure_storage(self):
        """Buat folder dan file storage jika belum ada."""
        os.makedirs(STORAGE_DIR, exist_ok=True)
        if not os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "w") as f:
                json.dump({}, f)

    def _load_all(self) -> Dict[str, Any]:
        """Muat semua encoding dari file JSON."""
        try:
            with open(ENCODINGS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_all(self, data: dict) -> Optional[str]:
        """Simpan semua encoding ke file JSON."""
        try:
            with open(ENCODINGS_FILE, "w") as f:
                json.dump(data, f)
            return None
        except Exception as e:
            return str(e)

    # ── Public Methods ────────────────────────────────────────────────────────

    def save_encoding(self, employee_id: str, encoding: List[float]) -> Optional[str]:
        """
        Simpan encoding wajah karyawan ke storage.

        Structure encodings.json:
        {
            "EMP001": { "encoding": [0.12, -0.34, ...] },
            "EMP002": { "encoding": [0.56,  0.78, ...] }
        }

        Returns:
            None  → sukses
            str   → pesan error
        """
        data = self._load_all()
        data[employee_id] = {"encoding": encoding}
        return self._save_all(data)

    def verify(
        self,
        employee_id: str,
        live_encoding: List[float]
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Bandingkan live encoding dengan encoding yang tersimpan.

        Returns:
            ({ match: bool, confidence: float }, None) → sukses
            (None, "error message")                    → gagal
        """
        data = self._load_all()

        # Cek apakah karyawan sudah enroll
        if employee_id not in data:
            return None, f"Employee {employee_id} has no enrolled face. Please enroll first"

        stored_encoding_list = data[employee_id].get("encoding")
        if not stored_encoding_list:
            return None, f"Encoding data for employee {employee_id} is corrupted"

        # Konversi list → numpy array
        stored_np = self.encoder.encoding_to_array(stored_encoding_list)
        live_np   = self.encoder.encoding_to_array(live_encoding)

        # Hitung face distance (0.0 = identik, 1.0 = berbeda total)
        distance = face_recognition.face_distance([stored_np], live_np)[0]

        # Konversi distance ke confidence score (0.0–1.0, makin tinggi makin mirip)
        # Formula: confidence = 1 - distance (linear, cukup untuk basic use case)
        confidence = round(float(1.0 - distance), 4)

        is_match = bool(distance <= FACE_DISTANCE_THRESHOLD)

        return {
            "match"     : is_match,
            "confidence": confidence,
            "distance"  : round(float(distance), 4)  # untuk debugging
        }, None

    def delete_encoding(self, employee_id: str) -> Tuple[bool, Optional[str]]:
        """
        Hapus data encoding karyawan.

        Returns:
            (True, None)    → sukses dihapus
            (False, str)    → employee tidak ditemukan atau error
        """
        data = self._load_all()

        if employee_id not in data:
            return False, f"Employee {employee_id} not found in storage"

        del data[employee_id]
        err = self._save_all(data)
        if err:
            return False, err

        return True, None

    def list_enrolled(self) -> List[str]:
        """Kembalikan list semua employee_id yang sudah punya encoding."""
        data = self._load_all()
        return list(data.keys())
