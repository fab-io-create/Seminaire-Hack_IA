"""
AI Workshop — Flask app with live inference streaming via SSE.
"""

import os, time, datetime, json, base64, threading, uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as transforms
from PIL import Image
# import face_recognition  # disabled: no webcam available
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUTS_DIR = "./outputs"
AUTHORIZED_FACES_DIR = "./authorized_faces"
MODELS_DIR = "./models"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(AUTHORIZED_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = {
    "fire_classifier":  os.path.join(MODELS_DIR, "model_classification.pth"),
    "fire_localizer":   os.path.join(MODELS_DIR, "model_localisation.pt"),
    "object_detector":  os.path.join(MODELS_DIR, "yolov8s.pt"),
}

app = Flask(__name__)

ts = lambda: datetime.datetime.now().strftime("%H:%M:%S")


def frame_to_b64jpg(frame, max_w=1280):
    """Encode a BGR frame to base64 JPEG, resized to max_w."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def sse_event(data):
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
# FACE RECOGNITION — DISABLED (no webcam available)
# Login is auto-validated as "Guest" so the dashboard opens immediately.
# To re-enable: uncomment `import face_recognition` above and this block,
# and restore `load_authorized_faces()` in the __main__ block.
# ═══════════════════════════════════════════════════════════════════════════════
# _auth_encodings = []
# _auth_names = []
#
#
# def load_authorized_faces():
#     global _auth_encodings, _auth_names
#     _auth_encodings.clear()
#     _auth_names.clear()
#     for fname in os.listdir(AUTHORIZED_FACES_DIR):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#         img = face_recognition.load_image_file(os.path.join(AUTHORIZED_FACES_DIR, fname))
#         encs = face_recognition.face_encodings(img)
#         if encs:
#             _auth_encodings.append(encs[0])
#             _auth_names.append(os.path.splitext(fname)[0].replace("_", " ").title())
#     print(f"Loaded {len(_auth_encodings)} authorized face(s): {_auth_names}")
#
#
# _camera = None
# _camera_lock = threading.Lock()
_logged_in_user = "Guest"  # auto-logged-in while face recognition is disabled
#
#
# def get_camera():
#     global _camera
#     with _camera_lock:
#         if _camera is None or not _camera.isOpened():
#             _camera = cv2.VideoCapture(0)
#     return _camera
#
#
# def release_camera():
#     global _camera
#     with _camera_lock:
#         if _camera and _camera.isOpened():
#             _camera.release()
#         _camera = None
#
#
# LOGIN_TIMEOUT = 30  # seconds
#
#
# def gen_login_frames():
#     global _logged_in_user
#     cam = get_camera()
#     frame_count = 0
#     t_start = time.time()
#     while True:
#         if _logged_in_user or (time.time() - t_start > LOGIN_TIMEOUT):
#             break
#         ret, frame = cam.read()
#         if not ret:
#             break
#         display = frame.copy()
#         if frame_count % 3 == 0:
#             small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
#             locs = face_recognition.face_locations(rgb_small)
#             encs = face_recognition.face_encodings(rgb_small, locs)
#             for (t, r, b, l), enc in zip(locs, encs):
#                 t, r, b, l = t*2, r*2, b*2, l*2
#                 if _auth_encodings:
#                     matches = face_recognition.compare_faces(_auth_encodings, enc, tolerance=0.5)
#                     dists = face_recognition.face_distance(_auth_encodings, enc)
#                     if any(matches):
#                         idx = int(np.argmin(dists))
#                         name = _auth_names[idx]
#                         cv2.rectangle(display, (l, t), (r, b), (0, 200, 0), 3)
#                         cv2.putText(display, f"{name} ({1-dists[idx]:.0%})", (l, t-12),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
#                         _logged_in_user = name
#                     else:
#                         cv2.rectangle(display, (l, t), (r, b), (0, 0, 200), 3)
#                         cv2.putText(display, "Unknown", (l, t-12),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
#         frame_count += 1
#         _, jpeg = cv2.imencode(".jpg", display)
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FIRE DETECTION — live SSE stream
# ═══════════════════════════════════════════════════════════════════════════════
CLASSIFICATION_CLASSES = ["fire", "no_fire", "start_fire"]
_cls_model = None
_yolo_fire = None

_fire_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _check_model_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {name} at {path}. Place the file in ./models/ or update MODELS dict in app.py."
        )


def load_fire_models():
    global _cls_model, _yolo_fire
    if _cls_model is None:
        _check_model_file(MODELS["fire_classifier"], "fire classifier")
        _check_model_file(MODELS["fire_localizer"], "fire localizer")
        from ultralytics import YOLO
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 3),
        )
        sd = torch.load(MODELS["fire_classifier"], map_location=DEVICE, weights_only=True)
        m.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")})
        m.to(DEVICE).eval()
        _cls_model = m
        _yolo_fire = YOLO(MODELS["fire_localizer"])
    return _cls_model, _yolo_fire


def gen_fire_sse(video_path):
    try:
        yield sse_event({"type": "log", "text": f"[{ts()}] Chargement des modèles de feu..."})
        cls_model, yolo_model = load_fire_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : impossible d'ouvrir la vidéo : {video_path}"})
            yield sse_event({"type": "done"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        yield sse_event({"type": "log", "text": f"[{ts()}] Vidéo : {w}x{h} @ {fps:.0f}ips, {total} images"})

        idx = 0
        cls_label, cls_conf = "no_fire", 0.0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t = _fire_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(cls_model(t)[0], dim=0)
            cls_label = CLASSIFICATION_CLASSES[probs.argmax().item()]
            cls_conf = probs.max().item()
            if cls_label != "no_fire":
                yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : {cls_label} ({cls_conf:.0%})"})

            if cls_label != "no_fire":
                results = yolo_model(frame, conf=0.3, verbose=False)
                annotated = results[0].plot()
                for box in results[0].boxes:
                    det = yolo_model.names[int(box.cls[0])]
                    yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : YOLO {det} ({float(box.conf[0]):.0%})"})
            else:
                annotated = frame.copy()

            color = (0, 0, 255) if cls_label == "fire" else (0, 165, 255) if cls_label == "start_fire" else (0, 255, 0)
            cv2.putText(annotated, f"{cls_label} ({cls_conf:.0%})", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if idx % 2 == 0:
                target_time = t_start + idx / fps
                wait = target_time - time.time()
                if wait > 0:
                    time.sleep(wait)
                yield sse_event({"type": "frame", "data": frame_to_b64jpg(annotated)})

            idx += 1

        cap.release()
        yield sse_event({"type": "log", "text": f"[{ts()}] Terminé — {idx} images traitées"})
    except Exception as e:
        yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : {e}"})
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
COCO_MAP = {"keys": None, "wallet": None, "watch": "clock", "remote": "remote", "smartphone": "cell phone"}
FR_NAMES = {"keys": "clés", "wallet": "portefeuille", "watch": "montre", "remote": "télécommande", "smartphone": "téléphone"}
_yolo_obj = None


def load_obj_model():
    global _yolo_obj
    if _yolo_obj is None:
        _check_model_file(MODELS["object_detector"], "object detector")
        from ultralytics import YOLO
        _yolo_obj = YOLO(MODELS["object_detector"])
    return _yolo_obj


def _friendly_name(cls_name):
    for k, v in COCO_MAP.items():
        if v == cls_name:
            return FR_NAMES.get(k, k)
    return cls_name


def gen_objects_sse(video_path, selected):
    try:
        yield sse_event({"type": "log", "text": f"[{ts()}] Chargement du modèle de détection d'objets..."})
        model = load_obj_model()

        target_coco = {COCO_MAP[o] for o in selected if COCO_MAP.get(o)}
        has_unmapped = [o for o in selected if not COCO_MAP.get(o)]
        selected_fr = [FR_NAMES.get(o, o) for o in selected]
        yield sse_event({"type": "log", "text": f"[{ts()}] Recherche de : {', '.join(selected_fr)}"})
        if has_unmapped:
            unmapped_fr = [FR_NAMES.get(o, o) for o in has_unmapped]
            yield sse_event({"type": "log", "text": f"[{ts()}] {', '.join(unmapped_fr)} non présents dans COCO — affichage de toutes les détections"})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : impossible d'ouvrir la vidéo : {video_path}"})
            yield sse_event({"type": "done"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        yield sse_event({"type": "log", "text": f"[{ts()}] Vidéo : {w}x{h} @ {fps:.0f}ips, {total} images"})

        idx = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25, verbose=False)
            annotated = frame.copy()

            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name in target_coco or (has_unmapped and conf > 0.3):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    friendly = _friendly_name(cls_name)
                    yield sse_event({"type": "log", "text": f"[{ts()}] Image {idx} ({idx/fps:.1f}s) : {friendly} ({conf:.0%})"})
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{friendly} {conf:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if idx % 2 == 0:
                target_time = t_start + idx / fps
                wait = target_time - time.time()
                if wait > 0:
                    time.sleep(wait)
                yield sse_event({"type": "frame", "data": frame_to_b64jpg(annotated)})

            idx += 1

        cap.release()
        yield sse_event({"type": "log", "text": f"[{ts()}] Terminé — {idx} images traitées"})
    except Exception as e:
        yield sse_event({"type": "log", "text": f"[{ts()}] ERREUR : {e}"})
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# Face-recognition login is disabled (no webcam). The dashboard opens
# immediately; /video_feed returns nothing, /check_login always succeeds.
@app.route("/video_feed")
def video_feed():
    return Response(b"", mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/check_login")
def check_login():
    return jsonify({"success": True, "name": _logged_in_user or "Guest"})


@app.route("/logout", methods=["POST"])
def logout():
    # No-op while face recognition is disabled — we stay auto-logged in.
    return jsonify({"ok": True})


# ── File upload endpoints (save file, return path token) ──
_uploads = {}  # uid -> {"path": str, "time": float}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    uid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(f.filename)[1]
    path = os.path.join(OUTPUTS_DIR, f"upload_{uid}{ext}")
    f.save(path)
    _uploads[uid] = {"path": path, "time": time.time()}
    # Clean up stale uploads (older than 10 minutes)
    stale = [k for k, v in _uploads.items() if time.time() - v["time"] > 600]
    for k in stale:
        try:
            os.remove(_uploads[k]["path"])
        except OSError:
            pass
        del _uploads[k]
    return jsonify({"id": uid})


# ── SSE streaming endpoints ──


def _pop_upload(uid):
    """Pop an uploaded file path by uid, or return None."""
    if uid:
        entry = _uploads.pop(uid, None)
        if entry:
            return entry["path"]
    return None


@app.route("/stream/fire")
def stream_fire():
    path = _pop_upload(request.args.get("upload")) or "data/foret.mp4"
    return Response(gen_fire_sse(path), mimetype="text/event-stream")


@app.route("/stream/objects")
def stream_objects():
    path = _pop_upload(request.args.get("upload")) or "data/object_detection.mp4"
    selected = request.args.get("selected", "keys,wallet,watch,remote,smartphone").split(",")
    return Response(gen_objects_sse(path, selected), mimetype="text/event-stream")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)


if __name__ == "__main__":
    # load_authorized_faces()  # disabled: no webcam available
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
