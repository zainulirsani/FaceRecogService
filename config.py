"""
config.py — Konfigurasi terpusat
=================================
Semua nilai yang mungkin berubah antara environment (dev/prod)
dikumpulkan di sini. Diambil dari environment variable,
dengan fallback ke nilai default untuk development lokal.
"""

import os

# ── Security ──────────────────────────────────────────────────────────────────
# Set ini di Render.com dashboard → Environment Variables
API_SECRET = os.environ.get("API_SECRET", "dev-secret-ganti-di-production")

# ── Storage ───────────────────────────────────────────────────────────────────
# Path file JSON penyimpan face encodings
# Di Render.com free tier, file ini ada di dalam container (ephemeral!)
# → Nanti saat migrasi ke VPS, path ini bisa diganti ke volume persisten
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR     = os.path.join(BASE_DIR, "storage")
ENCODINGS_FILE  = os.path.join(STORAGE_DIR, "encodings.json")

# ── Face Recognition ──────────────────────────────────────────────────────────
# Threshold jarak wajah: makin kecil = makin ketat
# 0.6 = default dlib, 0.5 = lebih ketat (recommended untuk absensi)
FACE_DISTANCE_THRESHOLD = float(os.environ.get("FACE_DISTANCE_THRESHOLD", "0.5"))

# Jumlah foto enroll per karyawan (untuk akurasi lebih baik, bisa > 1)
# Untuk sekarang: 1 foto cukup
MAX_ENCODINGS_PER_EMPLOYEE = int(os.environ.get("MAX_ENCODINGS_PER_EMPLOYEE", "1"))

# ── Anti-Spoofing ─────────────────────────────────────────────────────────────
# Threshold laplacian variance untuk deteksi blur (foto di layar cenderung blur)
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "80.0"))

# ── Image Processing ──────────────────────────────────────────────────────────
# Resize gambar sebelum diproses (hemat memory di Render.com)
MAX_IMAGE_WIDTH  = 640
MAX_IMAGE_HEIGHT = 640
