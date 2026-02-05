# server/receive_updates.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
UPDATES = ROOT / "server" / "updates"
PENDING = ROOT / "server" / "pending"
PENDING.mkdir(parents=True, exist_ok=True)

def main():
    pkg_files = list(UPDATES.glob("*_package.json"))
    if not pkg_files:
        print("No package json files found in server/updates/. Run clients first.")
        return
    for p in pkg_files:
        pkg = json.loads(p.read_text())
        wf = UPDATES / pkg["weights_file"]
        if not wf.exists():
            print(f"weights {wf.name} missing for {p.name}, skipping.")
            continue
        # move files to pending
        tgt_w = PENDING / wf.name
        tgt_p = PENDING / p.name
        wf.replace(tgt_w)
        p.replace(tgt_p)
        print(f"Moved {wf.name} & {p.name} -> pending/")
    print("Done. Run server/verify_hash.py then server/fedavg.py")

if __name__ == "__main__":
    main()
