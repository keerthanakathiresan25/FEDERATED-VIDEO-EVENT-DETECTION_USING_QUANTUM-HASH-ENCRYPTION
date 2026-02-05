1) Preprocess - create clips:
   python preprocess/prepare_clips.py

2) Train clients (simulate two clients):
   python client_1/train.py
   python client_2/train.py

3) Server receive:
   python server/receive_updates.py

4) Verify (optional):
   python server/verify_hash.py

5) Aggregate:
   python server/fedavg.py

6) Start UI and test:
   python web_ui/app.py
   Upload test video -> result shows label and confidence.

Notes:
- Ensure dataset/ has folders exact names matching client_common/classes.py.
- processed_dataset/ will be created by preprocessing.
- Keep epochs small (2) for demo; increase for better accuracy.
- Install dependencies: pip install torch torchvision opencv-python tqdm
