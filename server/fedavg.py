import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENDING = ROOT / "server" / "updates"   # <<< FIXED
OUT = ROOT / "global_model.pth"

def load_state(p):
    return torch.load(p, map_location="cpu")

def fed_avg(states):
    avg = {}
    n = len(states)
    keys = list(states[0].keys())
    for k in keys:
        acc = None
        for s in states:
            v = s[k].float()
            if acc is None:
                acc = v.clone()
            else:
                acc += v
        avg[k] = (acc / n).clone()
    return avg

def main():
    ups = list(PENDING.glob("*_update.pth"))
    if not ups:
        print("No updates in updates/ to aggregate.")
        return
    states = []
    for u in ups:
        print("Loading", u.name)
        states.append(load_state(u))
    print(f"Aggregating {len(states)} models...")
    avg = fed_avg(states)
    torch.save(avg, OUT)
    print("Saved aggregated global model to", OUT)

if __name__ == "__main__":
    main()
