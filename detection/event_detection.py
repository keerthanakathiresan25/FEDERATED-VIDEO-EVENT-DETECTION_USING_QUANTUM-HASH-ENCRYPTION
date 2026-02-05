# detection/event_detection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from pathlib import Path
import numpy as np
import cv2
import secrets
from torchvision.models.video import r3d_18, R3D_18_Weights
from client_common.classes import CLASSES

ROOT = Path(__file__).resolve().parent.parent
GLOBAL = ROOT / "global_model_finetuned.pth"  # ✅ Use fine-tuned model
FRAME_OUTPUT = ROOT / "web_ui" / "static" / "frames"
FRAME_OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
FRAMES_PER_CLIP = 16
TARGET_SIZE = (112, 112)

def build_model(num_classes):
    try:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    except:
        model = r3d_18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_global():
    if not GLOBAL.exists():
        raise FileNotFoundError("❌ global_model_finetuned.pth NOT FOUND.")
    sd = torch.load(GLOBAL, map_location="cpu")
    model = build_model(len(CLASSES))
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def extract_preview_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // num_frames, 1)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0 and len(frames) < num_frames:
            frame = cv2.resize(frame, (160, 120))
            fname = f"frame_{len(frames)+1}.jpg"
            fpath = FRAME_OUTPUT / fname
            cv2.imwrite(str(fpath), frame)
            frames.append(fname)
        count += 1
    cap.release()
    return frames

def video_to_clips(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, TARGET_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) < FRAMES_PER_CLIP:
        return []
    clips = []
    num = len(frames) // FRAMES_PER_CLIP
    for i in range(num):
        clip = np.array(frames[i*FRAMES_PER_CLIP:(i+1)*FRAMES_PER_CLIP], dtype=np.float32) / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))
        clips.append(clip)
    return clips

def detect_event(video_path):
    model = load_global()
    clips = video_to_clips(video_path)
    preview_frames = extract_preview_frames(video_path)

    if not clips:
        return {
            "event": "Video too short",
            "confidence": 0,
            "duration": "Unknown",
            "frames": [],
            "key": None
        }

    preds = []
    for clip in clips:
        x = torch.tensor(clip).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
            preds.append(p)

    avg = np.mean(preds, axis=0)
    idx = int(avg.argmax())
    event_name = CLASSES[idx]
    confidence = round(float(avg[idx]) * 100, 2)
    duration_seconds = len(clips) * (FRAMES_PER_CLIP / 30)
    encryption_key = secrets.token_hex(16)

    return {
        "event": event_name,
        "confidence": confidence,
        "duration": f"{duration_seconds:.1f} sec",
        "frames": preview_frames,
        "key": encryption_key
    }
