# client_1/train.py (Fast version)
import sys, os, json, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms

# -----------------------------
# CONFIGURATION
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed_dataset"
UPDATE_DIR = ROOT / "server" / "updates"
UPDATE_DIR.mkdir(parents=True, exist_ok=True)

from client_common.classes import CLASSES

CLIENT_NAME = "client_1"
CLIENT_CLASSES = ["jumping_exercise", "pushup_exercise", "basketball_play"]

EPOCHS = 3
BATCH_SIZE = 6            # ✅ bigger batch = fewer steps
LR = 5e-4
DEVICE = torch.device("cpu")

# -----------------------------
# DATASET
# -----------------------------
class ClipDataset(Dataset):
    def __init__(self, classes):
        self.samples = []
        self.class_to_idx = {c:i for i,c in enumerate(CLASSES)}

        for c in classes:
            folder = PROCESSED / c
            if not folder.exists():
                continue
            for f in folder.glob("*.npy"):
                self.samples.append((str(f), self.class_to_idx[c]))

        # ✅ limit dataset to 800 samples for faster training
        if len(self.samples) > 800:
            self.samples = random.sample(self.samples, 800)

        print(f"\n[{CLIENT_NAME}] ✅ Loaded {len(self.samples)} samples total")
        self.normalize = transforms.Normalize(mean=[0.45]*3, std=[0.225]*3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            clip = np.load(path, allow_pickle=False).astype(np.float32)/255.0
        except Exception:
            return self.__getitem__((idx+1) % len(self.samples))
        if clip.shape[-1] != 3:
            return self.__getitem__((idx+1) % len(self.samples))

        # ✅ use only 8 frames instead of all 16 (faster)
        clip = clip[::2][:8]

        clip = np.transpose(clip, (3,0,1,2))  # (C,T,H,W)
        tensor = torch.tensor(clip)
        for t in range(tensor.shape[1]):
            tensor[:,t] = self.normalize(tensor[:,t])
        return tensor, torch.tensor(label, dtype=torch.long)

# -----------------------------
# MODEL
# -----------------------------
def build_model(num_classes):
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)

    # ✅ Freeze early layers to train only classifier head
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------------
# TRAIN
# -----------------------------
def train():
    ds = ClipDataset(CLIENT_CLASSES)
    if len(ds) == 0:
        print("❌ No clips found.")
        return

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = build_model(len(CLASSES)).to(DEVICE)
    opt = optim.Adam(model.fc.parameters(), lr=LR)  # ✅ only classifier
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.85)

    print(f"\n[{CLIENT_NAME}] 🚀 Fast Training Started...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        print(f"Epoch {epoch+1}/{EPOCHS}")

        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            total += yb.size(0)
            correct += pred.eq(yb).sum().item()

            if (i+1) % 20 == 0:
                print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

        acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Accuracy: {acc:.2f}%\n")
        scheduler.step()

    # -----------------------------
    # SAVE UPDATE
    # -----------------------------
    update_path = UPDATE_DIR / f"{CLIENT_NAME}_update.pth"
    torch.save(model.state_dict(), update_path)

    import client_common.encrypt as enc
    pkg = enc.package_update(update_path, metadata={"client": CLIENT_NAME})
    with open(UPDATE_DIR / f"{CLIENT_NAME}_package.json", "w") as f:
        json.dump(pkg, f)

    print(f"\n[{CLIENT_NAME}] ✅ Model Saved → {update_path.name}")
    print(f"[{CLIENT_NAME}] 🔐 Hash → {pkg['hash']}\n")

# -----------------------------
if __name__ == "__main__":
    train()
