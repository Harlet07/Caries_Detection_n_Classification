"""Caries-only detection API built on the AlphaDent YOLOv8x-seg model.

Class ids 0-2 (Abrasion, Filling, Crown) are never requested from the model
and are dropped again defensively. Ids 3-8 map to G.V. Black Class I-VI.
Uploaded images live only in memory and are never written to disk or logged.
"""
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE.parent / "weights" / "yolov8x_g_v_black_classes_960px.pt"
FRONTEND = BASE.parent / "frontend"

MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 1920
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
CARIES = {3: "Class I", 4: "Class II", 5: "Class III",
          6: "Class IV", 7: "Class V", 8: "Class VI"}

Image.MAX_IMAGE_PIXELS = 120_000_000  # reject decompression bombs

log = logging.getLogger("caries")
state = {"model": None, "device": "cpu", "error": None}
infer_lock = Lock()  # one inference at a time; the model object is not thread-safe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup.
    try:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found at {WEIGHTS}")
        model = YOLO(str(WEIGHTS))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        state.update(model=model, device=device, error=None)
        log.warning("Model loaded on %s", device)
    except Exception as exc:  # keep the server up so /health and the UI still work
        state["error"] = str(exc)
        log.error("Model failed to load: %s", exc)
    yield


app = FastAPI(title="Caries detector", lifespan=lifespan)


def err(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_inference(img: Image.Image):
    # classes= filters ids 0-2 at inference time.
    return state["model"].predict(
        img, imgsz=960, conf=0.00, classes=list(CARIES),
        device=state["device"], verbose=False)[0]


def clip01(v: float) -> float:
    return round(min(1.0, max(0.0, float(v))), 4)


def parse(result) -> list:
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    cls = result.boxes.cls.tolist()
    conf = result.boxes.conf.tolist()
    boxes = result.boxes.xyxyn.tolist()
    # masks.xyn is index-aligned with boxes; it is None for non-segmentation output
    polys = result.masks.xyn if result.masks is not None else None
    for i, (c, p, b) in enumerate(zip(cls, conf, boxes)):
        c = int(c)
        if c not in CARIES:  # defensive filter
            continue
        poly = []
        if polys is not None and i < len(polys):
            poly = [[clip01(x), clip01(y)] for x, y in polys[i].tolist()]
        out.append({"label": CARIES[c], "conf": round(float(p), 4),
                    "polygon": poly, "box": [clip01(v) for v in b]})
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        return err(503, f"Model is not available: {state['error']}")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        return err(415, "Unsupported file type. Please upload a JPG, PNG or WEBP image.")

    data = file.file.read(MAX_BYTES + 1)  # read at most limit+1 bytes
    if len(data) > MAX_BYTES:
        return err(413, "File too large. Maximum size is 15 MB.")
    if not data:
        return err(400, "The uploaded file is empty.")

    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            return err(415, "Unsupported image format. Please upload a JPG, PNG or WEBP image.")
        img = ImageOps.exif_transpose(img).convert("RGB")  # honour phone rotation
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return err(400, "The file could not be read as a valid image.")
    finally:
        del data

    width, height = img.size
    try:
        with infer_lock:
            try:
                result = run_inference(img)
            except Exception as exc:
                if not is_oom(exc):
                    raise
                # GPU out of memory: free the cache and retry once on CPU.
                # The model stays on CPU afterwards so later requests don't fail the same way.
                torch.cuda.empty_cache()
                state["model"].to("cpu")
                state["model"].predictor = None  # force ultralytics to rebuild for the new device
                state["device"] = "cpu"
                result = run_inference(img)
        detections = parse(result)
    except Exception as exc:
        log.error("Inference failed: %s", type(exc).__name__)  # never log image content
        return err(500, "The analysis failed on the server. Please try again or use a smaller image.")

    return {"width": width, "height": height, "detections": detections}


# Mounted last so the API routes above take precedence.
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")