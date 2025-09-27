from flask import render_template, session, redirect, url_for, flash, request
from datetime import datetime

class Dashboard:
    def __init__(self, logs_collection, sentences_collection, search_history_collection):
        """
        Initialize Dashboard controller with required collections
        Args:
            logs_collection: MongoDB collection for logs
            sentences_collection: MongoDB collection for corpus sentences
            search_history_collection: MongoDB collection for user search history
        """
        self.logs = logs_collection
        self.sentences = sentences_collection
        self.search_history = search_history_collection

    def home(self):
        if "user_id" not in session:
            flash("Please log in to access the dashboard.", "warning")
            # ✅ Changed from url_for("accounts.login") to url_for("login")
            return redirect(url_for("login"))

        query = request.args.get("search")
        domain_filter = request.args.get("domain")
        user_email = session.get("user_email")

        # Fetch recent search history for the logged-in user
        recent_searches = list(self.search_history.find(
            {"user_email": user_email}
        ).sort("timestamp", -1).limit(5))

        # If there's a new search query, add it to history
        if query:
            search_entry = {
                "user_email": user_email,
                "query": query,
                "domain_filter": domain_filter,
                "timestamp": datetime.utcnow()
            }
            self.search_history.insert_one(search_entry)

        # Domain stats for the chart (only approved entries)
        domain_stats_cursor = self.sentences.aggregate([
            {"$match": {"status": "Approved"}},
            {"$unwind": "$domain"},
            {"$group": {"_id": "$domain", "count": {"$sum": 1}}}
        ])
        domain_stats = {d["_id"]: d["count"] for d in domain_stats_cursor}

        return render_template(
            "dashboard.html",
            query=query,
            recent_searches=recent_searches,
            selected_domain=domain_filter,
            domain_stats=domain_stats
        )