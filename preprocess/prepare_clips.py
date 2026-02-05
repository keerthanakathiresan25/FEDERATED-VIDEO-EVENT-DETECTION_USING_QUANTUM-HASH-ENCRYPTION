# preprocess/prepare_clips.py
"""
Create 16-frame clips from videos in dataset/<class>/
Outputs .npy clips to processed_dataset/<class>/*.npy

Run from project root:
python preprocess/prepare_clips.py
"""
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT / "dataset"
OUTPUT_ROOT = PROJECT / "processed_dataset"
FRAMES_PER_CLIP = 16
TARGET_SIZE = (112, 112)
SKIP_SMALL = True  # if video has fewer frames than FRAMES_PER_CLIP, skip

def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def extract_clips(video_path):
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
        if SKIP_SMALL:
            return []
        # pad with last frame
        while len(frames) < FRAMES_PER_CLIP:
            frames.append(frames[-1].copy())
    clips = []
    num = len(frames) // FRAMES_PER_CLIP
    for i in range(num):
        clip = np.array(frames[i*FRAMES_PER_CLIP:(i+1)*FRAMES_PER_CLIP], dtype=np.uint8)
        clips.append(clip)
    return clips

def main():
    ensure_dirs()
    classes = [d for d in DATASET_ROOT.iterdir() if d.is_dir()]
    if not classes:
        print("No class folders found in dataset/. Put your videos inside dataset/<class>/")
        return
    for cls in classes:
        out_cls = OUTPUT_ROOT / cls.name
        out_cls.mkdir(parents=True, exist_ok=True)
        videos = list(cls.glob("*"))
        saved = 0
        print(f"\nProcessing class '{cls.name}' - {len(videos)} videos")
        for v in tqdm(videos):
            try:
                clips = extract_clips(v)
                for i, clip in enumerate(clips):
                    fname = out_cls / f"{v.stem}_clip{i:03d}.npy"
                    np.save(fname, clip)
                    saved += 1
            except Exception as e:
                print(f"Skipped {v} due to {e}")
        print(f"Saved {saved} clips for class {cls.name}")

if __name__ == "__main__":
    main()
