import pytest
import tempfile
import os
from io import BytesIO

import datetime, io, csv, json, os, tempfile
from flask import render_template, request, redirect, url_for, flash, session, send_file, current_app
from openpyxl import Workbook
from werkzeug.utils import secure_filename
from Extractor import Extractor  # Assuming Extractor.py is in the app root
from translator import JSONTranslator  # Assuming translator.py is in the app root
# Import centralized database collections
from config.database import sentences_collection, user_contributions_collection
# Import centralized log_action
from utils.logging import log_action

load_dotenv()
SECRET_KEY = os.environ.get("FLASK_SECRET")
MONGO_URI = os.environ.get("MONGO_URI")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # Ensure Groq API key is loaded
# ----------------------------
# Database Setup (MongoDB Atlas)
# ----------------------
# ---------- DB init ----------
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable not set")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")


class Corpus:
    def __init__(self, sentences_collection, search_history_collection, log_action, upload_folder, temp_folder):
        self.sentences_collection = sentences_collection
        self.search_history_collection = search_history_collection
        self.log_action = log_action
        self.upload_folder = upload_folder
        self.temp_folder = temp_folder

        # Ensure upload and temp directories exist
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)

    def allowed_file(self, filename):
        # Use ALLOWED_EXTENSIONS from app config
        return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

    def corpus_stats(self):
        total_approved_entries = self.sentences_collection.count_documents({"status": "Approved"})
        domains_cursor = self.sentences_collection.distinct("domain", {"status": "Approved"})

        # Aggregate domain counts for approved sentences
        domain_counts_pipeline = [
            {"$match": {"status": "Approved"}},
            {"$unwind": "$domain"},
            {"$group": {"_id": "$domain", "count": {"$sum": 1}}}
        ]
        domain_counts_result = list(self.sentences_collection.aggregate(domain_counts_pipeline))
        domain_counts = {d["_id"]: d["count"] for d in domain_counts_result}

        # Placeholder word count logic (implement real word counting later if needed)
        word_count = 0
        for sentence in self.sentences_collection.find({"status": "Approved"}, {"zulu": 1, "english": 1}):
            word_count += len(sentence.get("zulu", "").split())
            word_count += len(sentence.get("english", "").split())

        stats = {
            "total_entries": total_approved_entries,
            "domains": list(domains_cursor),
            "domain_counts": domain_counts,
            "word_count": word_count
        }

        return render_template("corpus_stats.html", stats=stats)

    def upload(self):
        if request.method == "POST":
            if "file" not in request.files:
                flash("No file part", "danger")
                return redirect(request.url)

            file = request.files["file"]

            if file.filename == "":
                flash("No selected file", "warning")
                return redirect(request.url)

            if not self.allowed_file(file.filename):
                flash("File type not allowed", "danger")
                return redirect(request.url)

            filename = secure_filename(file.filename)
            filepath = os.path.join(self.upload_folder, filename)
            file.save(filepath)

            # Log the upload
            self.log_action("file_uploaded", details={"filename": filename})

            flash(f"File '{filename}' uploaded successfully!", "success")
            return redirect(url_for("dashboard"))

        return render_template("upload.html")