# server/verify_hash.py
import json
from pathlib import Path
import client_common.encrypt as enc

ROOT = Path(__file__).resolve().parent.parent
PENDING = ROOT / "server" / "pending"

def main():
    pkgs = list(PENDING.glob("*_package.json"))
    if not pkgs:
        print("No packages in pending.")
        return
    for p in pkgs:
        pkg = json.loads(p.read_text())
        wf = PENDING / pkg["weights_file"]
        ok = enc.verify_package(wf, pkg)
        print(p.name, "OK" if ok else "FAILED")
    print("Verification finished.")

if __name__ == "__main__":
    main()
