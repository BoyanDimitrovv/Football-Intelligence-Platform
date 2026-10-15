"""Football Intelligence Platform - Flask application entry point."""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from database.db import init_db


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "football.db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def create_app() -> Flask:
    """Create, configure and return the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    # Configuration is read from the environment where possible.
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev-only-not-a-real-secret"
    )
    app.config["DATABASE_PATH"] = str(DB_PATH)
    app.config["JSON_SORT_KEYS"] = False

    # Make sure the database directory and file exist (schema is added later).
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db(DB_PATH)

    _register_routes(app)
    _register_error_handlers(app)

    return app


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "football-intelligence-platform",
            }
        )


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("500.html"), 500


app = create_app()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug)
