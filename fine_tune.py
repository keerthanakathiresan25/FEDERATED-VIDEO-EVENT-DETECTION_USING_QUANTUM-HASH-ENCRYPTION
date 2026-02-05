import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
from tqdm import tqdm  # for progress bar

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
GLOBAL_MODEL = ROOT / "global_model.pth"
PROCESSED = ROOT / "processed_dataset"
OUT_PATH = ROOT / "global_model_finetuned.pth"

# -----------------------------
# Classes
# -----------------------------
from client_common.classes import CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1               # fine-tune just one epoch
BATCH_SIZE = 4           # higher batch for faster training
LR = 3e-5                # slightly smaller LR for precision

# -----------------------------
# Dataset
# -----------------------------
class CombinedDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                              std=[0.225, 0.225, 0.225])
        for c in CLASSES:
            folder = PROCESSED / c
            if not folder.exists():
                continue
            for f in folder.glob("*.npy"):
                self.samples.append((str(f), self.class_to_idx[c]))
        print(f"[FineTune] ✅ Loaded {len(self.samples)} clips total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            clip = np.load(path, allow_pickle=False).astype(np.float32) / 255.0
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        if clip.shape[-1] != 3:
            return self.__getitem__((idx + 1) % len(self.samples))
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C,T,H,W)
        tensor = torch.tensor(clip)
        for t in range(tensor.shape[1]):
            tensor[:, t] = self.normalize(tensor[:, t])
        return tensor, torch.tensor(label, dtype=torch.long)

# -----------------------------
# Build model (freeze layers)
# -----------------------------
def build_model(num_classes):
    try:
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
    except:
        model = r3d_18(pretrained=False)
    # freeze all but last few layers
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in list(model.layer4.named_parameters()) + list(model.fc.named_parameters()):
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------------
# Fine-tuning process
# -----------------------------
def fine_tune():
    if not GLOBAL_MODEL.exists():
        print("❌ global_model.pth not found. Run fedavg.py first.")
        return

    ds = CombinedDataset()
    if len(ds) == 0:
        print("❌ No processed dataset found.")
        return

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(GLOBAL_MODEL, map_location=DEVICE), strict=False)
    model.to(DEVICE)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader))
    criterion = nn.CrossEntropyLoss()

    print("\n[FineTune] 🚀 Fine-tuning started...\n")
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

        for xb, yb in loop:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            total += yb.size(0)
            correct += pred.eq(yb).sum().item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        acc = 100 * correct / total
        print(f"\n✅ Epoch {epoch+1} Done | Loss: {total_loss/len(loader):.4f} | Accuracy: {acc:.2f}%\n")

    torch.save(model.state_dict(), OUT_PATH)
    print(f"[FineTune] ✅ Saved fine-tuned model → {OUT_PATH.name}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    fine_tune()
