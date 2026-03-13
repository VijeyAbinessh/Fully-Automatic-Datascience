from flask import Flask, render_template, request, send_from_directory
import os
import subprocess
import uuid
import markdown
from flask import abort
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------- CLEANER ----------------
@app.route("/view/<filename>")
def view_file(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        abort(404)

    # Only allow markdown viewing
    if not filename.endswith(".md"):
        abort(403)

    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"]
    )

    return render_template(
        "view_md.html",
        content=html,
        filename=filename
    )

@app.route("/cleaner", methods=["GET", "POST"])
def cleaner():
    output_files = []
    if request.method == "POST":
        f = request.files["file"]
        uid = str(uuid.uuid4())
        in_path = os.path.join(UPLOAD_DIR, uid + "_" + f.filename)
        f.save(in_path)

        subprocess.run(
            ["python", os.path.join(FEATURE_DIR, "auto_cleaner.py"), in_path],
            cwd=OUTPUT_DIR
        )

        base = os.path.splitext(os.path.basename(in_path))[0]
        output_files = [
            f"cleaned_{base}.csv",
            f"cleaning_log_{base}.md"
        ]

    return render_template("cleaner.html", files=output_files)

# ---------------- REPORTER ----------------
@app.route("/reporter", methods=["GET", "POST"])
def reporter():
    files = []

    if request.method == "POST":
        f = request.files["file"]
        uid = str(uuid.uuid4())

        upload_path = os.path.join(UPLOAD_DIR, uid + "_" + f.filename)
        f.save(upload_path)

        # Run report generator
        subprocess.run(
            ["python", os.path.join(FEATURE_DIR, "auto_report.py"), upload_path],
            cwd=OUTPUT_DIR
        )

        # 🔥 FIX 1: detect generated files dynamically
        base = os.path.splitext(os.path.basename(upload_path))[0]

        files = [
            file for file in os.listdir(OUTPUT_DIR)
            if file.endswith(".md") and base in file
        ]

    return render_template("reporter.html", files=files)


# ---------------- TRAINER ----------------
@app.route("/trainer", methods=["GET", "POST"])
def trainer():
    columns = []
    model_file = None
    metrics = {}

    if request.method == "POST":

        # -------- STAGE 1: FILE UPLOAD --------
        if "upload" in request.form:
            file = request.files["file"]
            path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(path)

            df = pd.read_csv(path)
            columns = df.columns.tolist()

            return render_template(
                "trainer.html",
                columns=columns,
                file_path=path
            )

        # -------- STAGE 2: TRAIN MODEL --------
        elif "train" in request.form:
            path = request.form["file_path"]
            df = pd.read_csv(path)

            target = request.form.get("target")
            features = request.form.getlist("features")
            task = request.form.get("task")
            model_choice = request.form.get("model")

            from features.auto_trainer import train_model_ui

            model_file, metrics = train_model_ui(
                df, target, features, task, model_choice
            )

            return render_template(
                "trainer.html",
                columns=df.columns.tolist(),
                model_file=model_file,
                metrics=metrics
            )

    return render_template("trainer.html")


# ---------------- RESOURCE FINDER ----------------
@app.route("/resources", methods=["GET", "POST"])
def resources():
    result = None

    if request.method == "POST":
        problem = request.form["problem"]

        import importlib
        from features import resource_finder
        importlib.reload(resource_finder)

        result = resource_finder.find_resources_ui(problem)
        import json
        print("🔥 RESOURCE FINDER RESULT:")
        print(json.dumps(result, indent=2))

    return render_template("resources.html", result=result)


# ---------------- DOWNLOAD ----------------
@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
