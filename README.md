# Caries Detection Prototype

Upload a dental photo and get it back labeled with caries detections only (G.V. Black Class I–VI).
Backend: FastAPI + ultralytics (AlphaDent YOLOv8x-seg). Frontend: plain HTML/CSS/JS.

> Research prototype. Not a medical device and not a diagnosis. Results must be reviewed by a qualified dentist.

## Class mapping

The model has 9 classes. Ids 0–2 (Abrasion, Filling, Crown) are never requested and never returned.

| Model id | Model name | Shown as |
|---|---|---|
| 3 | Caries 1 | Class I |
| 4 | Caries 2 | Class II |
| 5 | Caries 3 | Class III |
| 6 | Caries 4 | Class IV |
| 7 | Caries 5 | Class V |
| 8 | Caries 6 | Class VI |

## Setup

1. Use Python 3.11.
2. Put the weights file at `backend/weights/yolov8x_AlphaDent_9_classes_960px.pt`
   (create the `weights` folder if it does not exist).
3. Install dependencies (run from the `project/` folder):
   ```
   pip install -r backend/requirements.txt
   ```
4. Run:
   ```
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
5. Open http://localhost:8000

`GET /health` returns `{"status":"ok"}`. If the weights are missing, the server still starts, and
`/predict` returns a 503 error that says so.

## API

`POST /predict` (multipart/form-data, field `file`; jpg/jpeg/png/webp, max 15 MB) returns:

```json
{"width": 1920, "height": 1280,
 "detections": [{"label": "Class III", "conf": 0.87,
                 "polygon": [[0.41, 0.52], [0.43, 0.55]],
                 "box": [0.40, 0.50, 0.45, 0.57]}]}
```

Coordinates are normalized to 0–1. The image is rotated per EXIF and downscaled to at most 1920 px
on its longest side before inference. Uploads are processed in memory only and are never stored or logged.

## Deployment notes

- **Performance:** yolov8x on CPU takes several seconds per image. Use a GPU host for production,
  or fine-tune and use a smaller model (e.g. yolov8s/m-seg). On GPU out-of-memory, the server
  retries once on CPU and stays on CPU afterwards.
- **HTTPS and reverse proxy:** these are medical photos, so serve over HTTPS behind a reverse proxy
  (nginx, Caddy, Traefik) and do not expose uvicorn directly to the internet.
- **Upload limit:** the app rejects files over 15 MB, but also set a limit at the proxy
  (nginx: `client_max_body_size 15m;`) so oversized requests are rejected before reaching Python.
- Add authentication and rate limiting if the service is public. Run a single worker per GPU.