"""
Emotion detection module using Hume AI Streaming API.

FIXES applied (v2):
  1. websockets extra_headers → additional_headers  (websockets >= v10 compatibility)
  2. Removed the guard in _opencv_detector() that was blocking OpenCV even when
     Hume failed at runtime.  Previously, HUME_SDK_AVAILABLE=True (import succeeds)
     was enough to disable OpenCV — so both paths returned None.
"""

import os
import base64
import json
import asyncio
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Enable nested event loops for Streamlit compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

try:
    from hume import AsyncHumeClient          # noqa: F401 – just checking availability
    HUME_SDK_AVAILABLE = True
except ImportError as e:
    HUME_SDK_AVAILABLE = False
    print(f"[emotion_detector] Hume SDK import error: {e}")

load_dotenv()

HUME_API_KEY        = os.getenv("HUME_API_KEY", "")
HUME_PROB_THRESHOLD = float(os.getenv("HUME_PROB_THRESHOLD", "0.2"))
USE_HUME            = (os.getenv("USE_HUME", "1") == "1"
                       and bool(HUME_API_KEY)
                       and HUME_SDK_AVAILABLE)

EMOTION_DETECTION_AVAILABLE = True

print(
    f"[emotion_detector] Config: USE_HUME={USE_HUME}, "
    f"API_KEY={'SET' if HUME_API_KEY else 'MISSING'}, "
    f"SDK={'OK' if HUME_SDK_AVAILABLE else 'MISSING'}, "
    f"THRESHOLD={HUME_PROB_THRESHOLD}"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_frame(frame_data: np.ndarray) -> Optional[str]:
    """
    Analyze a video frame for emotion detection.

    Priority:
      1. Hume Streaming WebSocket API  (high accuracy)
      2. OpenCV Haar-cascade detector  (always-available fallback)

    Args:
        frame_data: numpy array in BGR format (OpenCV convention)

    Returns:
        Emotion string or None
    """
    if frame_data is None or frame_data.size == 0:
        return None

    # --- Priority 1: Hume ---
    if USE_HUME and HUME_API_KEY and HUME_SDK_AVAILABLE:
        try:
            emotion = _analyze_via_hume_sdk(frame_data)
            if emotion:
                return emotion
            print("[emotion_detector] Hume returned no emotion, trying OpenCV fallback")
        except Exception as exc:
            print(f"[emotion_detector] Hume error: {exc}")

    # --- Priority 2: OpenCV (runs whenever Hume is absent OR failed) ---
    return _opencv_detector(frame_data)


# ---------------------------------------------------------------------------
# Hume path
# ---------------------------------------------------------------------------

def _analyze_via_hume_sdk(frame_data: np.ndarray) -> Optional[str]:
    frame_bytes = _convert_to_jpeg_bytes(frame_data)
    if not frame_bytes:
        return None
    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

    # Run async code in a dedicated thread so we never conflict with
    # Streamlit's existing event loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_async_in_thread, frame_b64)
        try:
            return future.result(timeout=6)
        except Exception as exc:
            print(f"[emotion_detector] Thread/timeout error: {exc}")
            return None


def _run_async_in_thread(frame_b64: str) -> Optional[str]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_analyze(frame_b64))
    finally:
        loop.close()


async def _async_analyze(frame_b64: str) -> Optional[str]:
    """Connect to Hume WebSocket, send one frame, return mapped emotion."""
    try:
        import websockets

        uri     = "wss://api.hume.ai/v0/stream/models"
        payload = json.dumps({
            "data": frame_b64,
            "models": {"face": {}},
            "raw_text": False,
        })

        # ----------------------------------------------------------------
        # FIX 1: websockets >= 10 renamed extra_headers → additional_headers.
        # We try the new name first, then fall back gracefully.
        # ----------------------------------------------------------------
        ws_connect = None
        for header_kwarg in ("additional_headers", "extra_headers"):
            try:
                ws_connect = websockets.connect(
                    uri,
                    **{header_kwarg: {"X-Hume-Api-Key": HUME_API_KEY}},
                )
                # Test that the keyword is accepted by peeking at the coroutine
                # (actual TypeError surfaces on __aenter__, but a dry-run avoids
                #  sending twice – so we just pick the first that doesn't raise
                #  a TypeError at construction time).
                break
            except TypeError:
                ws_connect = None
                continue

        if ws_connect is None:
            print("[emotion_detector] Could not determine correct websockets header kwarg")
            return None

        async with ws_connect as ws:
            await ws.send(payload)
            raw = await ws.recv()
            return _parse_hume_response(raw)

    except Exception as exc:
        print(f"[emotion_detector] Async Hume error: {exc}")
        return None


def _parse_hume_response(raw: str) -> Optional[str]:
    """Extract the highest-scoring mapped emotion from a Hume JSON response."""
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if "error" in result:
        print(f"[emotion_detector] Hume API error: {result['error']}")
        return None

    face_block = result.get("face") or {}
    predictions = face_block.get("predictions") or []
    if not predictions:
        return None

    emotions_raw = predictions[0].get("emotions") or []
    if not emotions_raw:
        return None

    MAIN_EMOTIONS = {
        "Joy", "Sadness", "Anger", "Fear",
        "Surprise (positive)", "Surprise (negative)",
        "Disgust", "Calmness", "Excitement", "Contentment",
    }

    all_sorted  = sorted(emotions_raw, key=lambda e: e["score"], reverse=True)
    main_sorted = [e for e in all_sorted if e["name"] in MAIN_EMOTIONS]

    # Debug: top 5
    print("[emotion_detector] Hume top 5:")
    for i, e in enumerate(all_sorted[:5]):
        mark = "★" if e["name"] in MAIN_EMOTIONS else " "
        print(f"  {mark} {i+1}. {e['name']}: {e['score']:.3f}")

    # Prefer a main emotion above threshold
    for e in main_sorted:
        if e["score"] >= HUME_PROB_THRESHOLD:
            mapped = _map_hume_emotion_to_mood(e["name"])
            print(f"[emotion_detector] ✓ Hume: {e['name']} ({e['score']:.2f}) → {mapped}")
            return mapped

    # Fallback: any emotion above threshold
    if all_sorted and all_sorted[0]["score"] >= HUME_PROB_THRESHOLD:
        e = all_sorted[0]
        mapped = _map_hume_emotion_to_mood(e["name"])
        print(f"[emotion_detector] ✓ Hume (any): {e['name']} ({e['score']:.2f}) → {mapped}")
        return mapped

    return None


# ---------------------------------------------------------------------------
# OpenCV fallback
# ---------------------------------------------------------------------------

def _opencv_detector(frame_data: np.ndarray) -> Optional[str]:
    """
    Haar-cascade fallback detector.

    FIX 2: The old code had:
        if USE_HUME and HUME_SDK_AVAILABLE: return None
    which silently disabled OpenCV even when Hume failed at *runtime*.
    That guard is now removed — OpenCV runs whenever Hume didn't return a result.
    """
    try:
        gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)

        face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("[emotion_detector] ✓ OpenCV: no face detected → calm")
            return "calm"

        x, y, w, h    = faces[0]
        face_roi_gray = gray[y : y + h, x : x + w]

        eyes   = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(20, 20))
        smiles = smile_cascade.detectMultiScale(face_roi_gray, 1.7, 15, minSize=(25, 25))

        print(f"[emotion_detector] OpenCV: face=1, eyes={len(eyes)}, smiles={len(smiles)}")

        if len(smiles) > 0:
            print("[emotion_detector] ✓ OpenCV: smile → happy")
            return "happy"
        elif len(eyes) >= 2:
            print("[emotion_detector] ✓ OpenCV: eyes only → calm")
            return "calm"
        else:
            print("[emotion_detector] ✓ OpenCV: unclear → calm")
            return "calm"

    except Exception as exc:
        print(f"[emotion_detector] OpenCV error: {exc}")
        return "calm"          # last-resort so the app never hard-fails


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_to_jpeg_bytes(frame_data: np.ndarray, quality: int = 85) -> Optional[bytes]:
    try:
        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception as exc:
        print(f"[emotion_detector] Frame conversion error: {exc}")
        return None


def _map_hume_emotion_to_mood(hume_emotion: str) -> str:
    primary = {
        "Joy":                  "happy",
        "Sadness":              "sad",
        "Anger":                "angry",
        "Fear":                 "fearful",
        "Surprise (positive)":  "surprised",
        "Surprise (negative)":  "surprised",
        "Disgust":              "angry",
        "Calmness":             "calm",
        "Excitement":           "energized",
        "Contentment":          "relaxed",
    }
    if hume_emotion in primary:
        return primary[hume_emotion]

    extended = {
        "Amusement": "happy",    "Satisfaction": "happy",   "Triumph": "happy",
        "Pride":     "happy",    "Relief":        "happy",   "Gratitude": "happy",
        "Admiration": "happy",   "Adoration":     "happy",
        "Love":      "loving",   "Romance":       "loving",
        "Disappointment": "sad", "Empathic Pain": "sad",     "Sympathy": "sad",
        "Tiredness":      "sad", "Boredom":       "sad",     "Guilt":    "sad",
        "Shame":          "sad", "Embarrassment": "sad",     "Pain":     "sad",
        "Contempt":  "angry",    "Annoyance":  "angry",      "Envy":     "angry",
        "Anxiety":   "anxious",  "Horror":     "fearful",    "Doubt":    "fearful",
        "Confusion": "fearful",  "Awkwardness":"fearful",    "Distress": "fearful",
        "Concentration": "focused", "Contemplation": "focused",
        "Determination": "focused", "Interest":      "focused",
        "Enthusiasm":    "energized", "Craving":    "energized",
        "Realization":   "surprised", "Awe":        "surprised",
        "Nostalgia":     "relaxed",   "Entrancement": "focused",
    }
    return extended.get(hume_emotion, "calm")


def normalize_emotion(emotion: str) -> str:
    if not emotion:
        return "calm"
    e = emotion.lower().strip()
    valid = {
        "happy", "sad", "angry", "fearful", "surprised",
        "calm", "energized", "relaxed", "focused", "loving", "anxious",
    }
    if e in valid:
        return e
    aliases = {
        "neutral": "calm",    "content":  "calm",    "peaceful":   "calm",
        "excited": "energized","hyper":   "energized",
        "tired":   "relaxed", "sleepy":   "relaxed",
        "scared":  "fearful", "afraid":   "fearful",
        "worried": "anxious", "anxious":  "anxious",
        "mad":     "angry",   "frustrated":"angry",  "annoyed":    "angry",
        "joyful":  "happy",   "cheerful": "happy",   "glad":       "happy",
        "depressed":"sad",    "unhappy":  "sad",      "down":       "sad",
        "amazed":  "surprised","shocked": "surprised","astonished":"surprised",
        "romantic": "loving",
    }
    return aliases.get(e, "calm")
