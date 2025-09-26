from flask import render_template, session, flash, redirect, url_for

# Moved basic_security_scan out of class if it's a utility function
def basic_security_scan(zulu, english, domain):
    # Example: block dangerous input
    blocked_words = ["<script>", "DROP TABLE", "DELETE FROM"]
    for word in blocked_words:
        if word.lower() in zulu.lower() or word.lower() in english.lower():
            return False
    return True

class Contributions:
    def __init__(self, sentences_collection, log_action):
        self.sentences = sentences_collection # Assuming contributions are stored in sentences
        self.log_action = log_action

    def contributions(self): # Renamed from `list_contributions` to `contributions` to match original
        user_email = session.get("user_email")
        if not user_email:
            flash("You need to be logged in to view contributions.", "warning")
            return redirect(url_for("accounts.login")) # Redirect if not logged in

        user_contributions = list(self.sentences.find({"added_by": user_email}).sort("created_at", -1))
        stats = {
            "total": len(user_contributions),
            "approved": sum(1 for c in user_contributions if c.get("status") == "Approved"),
            "pending": sum(1 for c in user_contributions if c.get("status") == "Pending"),
            "rejected": sum(1 for c in user_contributions if c.get("status") == "Rejected")
        }
        self.log_action("view_contributions", user=user_email)
        return render_template("contributions.html", contributions=user_contributions, stats=stats)