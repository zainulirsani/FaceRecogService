"""
Face Recognition Service - Entry Point
=======================================
Flask API yang menjadi jembatan antara Laravel dan fitur face recognition.
Dijalankan di Render.com / Railway sebagai microservice terpisah.
"""

from flask import Flask, request, jsonify
from functools import wraps
import os
import logging

# Import service layer (akan kita buat berikutnya)
from services.face_encoder import FaceEncoder
from services.face_matcher import FaceMatcher
from services.anti_spoof import AntiSpoof
from utils.image_utils import decode_base64_image
from utils.response import success_response, error_response

# ── Setup Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Init App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Init Services (singleton, load sekali saat startup) ──────────────────────
encoder  = FaceEncoder()
matcher  = FaceMatcher()
spoofer  = AntiSpoof()

# ── Auth Middleware ───────────────────────────────────────────────────────────
API_SECRET = os.environ.get("API_SECRET", "ganti-dengan-secret-kuat")

def require_auth(f):
    """
    Decorator: semua endpoint wajib pakai Bearer token.
    Laravel menyimpan secret ini di .env → FACE_API_SECRET
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return error_response("Unauthorized: missing token", 401)
        token = auth_header.split(" ", 1)[1]
        if token != API_SECRET:
            return error_response("Unauthorized: invalid token", 401)
        return f(*args, **kwargs)
    return decorated


# ── Helper: validasi request body ────────────────────────────────────────────
def get_required_fields(data: dict, fields: list):
    """Cek field wajib ada di request body, return error string jika kurang."""
    missing = [f for f in fields if not data.get(f)]
    return f"Missing fields: {', '.join(missing)}" if missing else None


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Health Check
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def health_check():
    """
    Dipakai Render.com untuk memastikan service hidup.
    Laravel juga bisa ping endpoint ini sebelum kirim request.
    """
    return success_response("Face Recognition Service is running", {
        "version": "1.0.0",
        "status": "healthy"
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Enroll
#  Dipanggil Laravel saat: admin mendaftarkan foto wajah karyawan baru
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/enroll", methods=["POST"])
@require_auth
def enroll():
    """
    Menerima foto karyawan, generate face encoding, simpan ke storage.

    Request Body (JSON):
        employee_id  : string  → ID unik karyawan (dari Laravel)
        image_base64 : string  → foto wajah format base64 (JPEG/PNG)

    Response:
        200 → { success: true, message: "enrolled" }
        400 → { success: false, message: "..." }
    """
    data = request.get_json(silent=True) or {}

    # Validasi input
    err = get_required_fields(data, ["employee_id", "image_base64"])
    if err:
        return error_response(err, 400)

    employee_id = str(data["employee_id"])
    logger.info(f"[ENROLL] employee_id={employee_id}")

    # 1. Decode base64 → numpy image array
    image, decode_err = decode_base64_image(data["image_base64"])
    if decode_err:
        return error_response(f"Image decode failed: {decode_err}", 400)

    # 2. Encode wajah → 128-dim vector
    encoding, encode_err = encoder.encode(image)
    if encode_err:
        return error_response(f"Face encoding failed: {encode_err}", 422)

    # 3. Simpan encoding ke storage
    save_err = matcher.save_encoding(employee_id, encoding)
    if save_err:
        return error_response(f"Storage error: {save_err}", 500)

    logger.info(f"[ENROLL] SUCCESS employee_id={employee_id}")
    return success_response("Face enrolled successfully", {
        "employee_id": employee_id
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Verify
#  Dipanggil Laravel saat: karyawan melakukan absensi
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/verify", methods=["POST"])
@require_auth
def verify():
    """
    Mencocokkan foto absen dengan encoding yang sudah tersimpan.

    Request Body (JSON):
        employee_id  : string  → ID karyawan yang sedang absen
        image_base64 : string  → foto selfie saat absen (base64)

    Response:
        200 → {
            success     : true,
            match       : true/false,
            confidence  : 0.0–1.0,   ← makin tinggi makin mirip
            spoof       : true/false, ← true = terdeteksi foto palsu
            message     : "..."
        }
    """
    data = request.get_json(silent=True) or {}

    err = get_required_fields(data, ["employee_id", "image_base64"])
    if err:
        return error_response(err, 400)

    employee_id = str(data["employee_id"])
    logger.info(f"[VERIFY] employee_id={employee_id}")

    # 1. Decode base64 → numpy image array
    image, decode_err = decode_base64_image(data["image_base64"])
    if decode_err:
        return error_response(f"Image decode failed: {decode_err}", 400)

    # 2. Anti-spoofing check (cegah foto dicetak/layar)
    is_spoof, spoof_reason = spoofer.check(image)
    if is_spoof:
        logger.warning(f"[VERIFY] SPOOF DETECTED employee_id={employee_id} reason={spoof_reason}")
        return success_response("Spoof detected", {
            "match"      : False,
            "confidence" : 0.0,
            "spoof"      : True,
            "reason"     : spoof_reason
        })

    # 3. Encode wajah dari foto absen
    live_encoding, encode_err = encoder.encode(image)
    if encode_err:
        return error_response(f"Face encoding failed: {encode_err}", 422)

    # 4. Load encoding tersimpan & bandingkan
    match_result, match_err = matcher.verify(employee_id, live_encoding)
    if match_err:
        return error_response(f"Match error: {match_err}", 422)

    logger.info(
        f"[VERIFY] employee_id={employee_id} "
        f"match={match_result['match']} "
        f"confidence={match_result['confidence']:.2f}"
    )

    return success_response("Verification complete", {
        "match"      : match_result["match"],
        "confidence" : match_result["confidence"],
        "spoof"      : False
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: Delete Enrollment
#  Dipanggil Laravel saat: karyawan resign / off-boarding
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/delete/<employee_id>", methods=["DELETE"])
@require_auth
def delete_enrollment(employee_id):
    """
    Hapus data encoding wajah karyawan dari storage.

    URL Param:
        employee_id : string → ID karyawan yang akan dihapus

    Response:
        200 → { success: true, message: "deleted" }
        404 → { success: false, message: "not found" }
    """
    logger.info(f"[DELETE] employee_id={employee_id}")

    deleted, err = matcher.delete_encoding(str(employee_id))
    if err:
        return error_response(err, 404)

    logger.info(f"[DELETE] SUCCESS employee_id={employee_id}")
    return success_response("Enrollment deleted successfully", {
        "employee_id": employee_id
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT: List Enrolled (opsional, untuk debug / admin panel)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/enrolled", methods=["GET"])
@require_auth
def list_enrolled():
    """
    Mengembalikan daftar employee_id yang sudah punya encoding tersimpan.
    Berguna untuk admin mengecek siapa yang sudah/belum enroll.
    """
    ids = matcher.list_enrolled()
    return success_response("Enrolled employees", {
        "count"        : len(ids),
        "employee_ids" : ids
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Render.com otomatis inject PORT env variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
