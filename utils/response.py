"""
utils/response.py — Standar Format JSON Response
==================================================
Semua response API menggunakan format yang sama
supaya Laravel mudah parsing-nya secara konsisten.
"""

from flask import jsonify
from typing import Any, Optional


def success_response(message: str, data: Optional[Any] = None, status_code: int = 200):
    """
    Format response sukses.

    Contoh output:
    {
        "success": true,
        "message": "Face enrolled successfully",
        "data": {
            "employee_id": "123"
        }
    }
    """
    payload = {
        "success": True,
        "message": message,
    }
    if data is not None:
        payload["data"] = data

    return jsonify(payload), status_code


def error_response(message: str, status_code: int = 400):
    """
    Format response error.

    Contoh output:
    {
        "success": false,
        "message": "Missing fields: employee_id",
        "data": null
    }
    """
    payload = {
        "success": False,
        "message": message,
        "data"   : None
    }
    return jsonify(payload), status_code
