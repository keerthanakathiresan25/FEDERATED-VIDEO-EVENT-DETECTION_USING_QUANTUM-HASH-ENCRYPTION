import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, redirect
from pathlib import Path
from detection.event_detection import detect_event
from web_ui.database import init_db, insert_result, get_all_results  # ✅ Database functions

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Initialize DB once
init_db()
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return redirect("/")
    file = request.files["video"]
    if file.filename == "":
        return redirect("/")

    video_path = UPLOAD_FOLDER / file.filename
    file.save(video_path)

    # Run detection
    result = detect_event(video_path)

    # ✅ Save detection result to DB
    insert_result(file.filename, result)

    return render_template("result.html", result=result, video_filename=file.filename)

# ✅ Optional route to view saved history
@app.route("/history")
def history():
    data = get_all_results()
    return render_template("history.html", results=data)

if __name__ == "__main__":
    app.run(debug=True)
