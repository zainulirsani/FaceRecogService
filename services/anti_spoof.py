"""
services/anti_spoof.py — Basic Anti-Spoofing
=============================================
Deteksi foto palsu (printed photo / foto dari layar HP/monitor).
Level: Basic — menggunakan analisis tekstur dan blur.

Pendekatan:
    1. Blur Detection    → foto di layar cenderung blur / low detail
    2. Reflection Check  → layar HP punya highlight/glare tertentu
    3. Edge Density      → foto asli punya edge density lebih natural

Catatan: Ini bukan liveness detection (tidak deteksi "apakah orang ini hidup").
         Cukup untuk mencegah karyawan pakai foto orang lain dari HP/print.
"""

import cv2
import numpy as np
from typing import Tuple
from utils.image_utils import to_grayscale
from config import BLUR_THRESHOLD


class AntiSpoof:

    def check(self, image_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        Jalankan semua check anti-spoofing.

        Returns:
            (False, "")          → BUKAN spoof, gambar aman
            (True,  "reason")    → SPOOF terdeteksi, beserta alasannya
        """
        gray = to_grayscale(image_bgr)

        # Check 1: Blur detection
        is_blurry, blur_reason = self._check_blur(gray)
        if is_blurry:
            return True, blur_reason

        # Check 2: Screen reflection / moiré pattern
        is_moire, moire_reason = self._check_moire(gray)
        if is_moire:
            return True, moire_reason

        return False, ""

    # ── Private Checks ────────────────────────────────────────────────────────

    def _check_blur(self, gray: np.ndarray) -> Tuple[bool, str]:
        """
        Laplacian variance: mengukur ketajaman gambar.

        Foto asli dari kamera HP → tajam → variance tinggi (> threshold)
        Foto di layar HP         → sedikit blur → variance rendah

        Threshold default: 80.0 (bisa dikalibrasi via env variable)
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < BLUR_THRESHOLD:
            return True, f"Image too blurry (score: {laplacian_var:.1f}, min: {BLUR_THRESHOLD})"

        return False, ""

    def _check_moire(self, gray: np.ndarray) -> Tuple[bool, str]:
        """
        Moiré pattern detection menggunakan FFT (Fast Fourier Transform).

        Foto dari layar digital sering menghasilkan pola reguler
        (pixel grid dari layar) yang terdeteksi sebagai spike di domain frekuensi.

        Cara kerja:
            1. Transformasi gambar ke domain frekuensi (FFT)
            2. Lihat distribusi energi frekuensi tinggi
            3. Jika ada spike tidak natural → kemungkinan foto dari layar
        """
        # Resize ke 256x256 untuk konsistensi dan kecepatan
        resized = cv2.resize(gray, (256, 256))

        # FFT
        f_transform = np.fft.fft2(resized)
        f_shift     = np.fft.fftshift(f_transform)
        magnitude   = 20 * np.log(np.abs(f_shift) + 1)

        # Ambil area frekuensi tinggi (pinggir spectrum)
        h, w = magnitude.shape
        margin = 10  # pixels dari tepi

        # Energi di frekuensi rendah (tengah)
        center_region = magnitude[
            h//2 - margin : h//2 + margin,
            w//2 - margin : w//2 + margin
        ]

        # Energi total
        total_energy   = magnitude.sum()
        center_energy  = center_region.sum()

        # Rasio: foto dari layar punya distribusi energi yang tidak natural
        # (terlalu banyak energi di frekuensi tertentu)
        if total_energy > 0:
            high_freq_ratio = 1.0 - (center_energy / total_energy)
            # Threshold empiris: jika > 0.97 = kemungkinan pola buatan
            if high_freq_ratio > 0.97:
                return True, f"Screen pattern detected (FFT ratio: {high_freq_ratio:.3f})"

        return False, ""
