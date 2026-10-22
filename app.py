"""Football Intelligence Platform - Flask application entry point.

A scouting platform that combines a football database, a from-scratch machine
learning model and an AI assistant to recommend players for a club's needs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from assistant.bot import ScoutingAssistant, llm_available
from database import repository as repo
from database.db import DEFAULT_DB_PATH, get_connection, init_db
from ml.model import ModelBundle
from ml.recommender import (
    POSITION_LABELS,
    POSITION_WEIGHTS,
    rank_players,
    score_player,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "football.db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "ml" / "model.json"


# ------------------------------------------------------------- bootstrap ----
def _ensure_ready(db_path: Path) -> None:
    """Make sure the database is seeded and a trained model is available."""
    init_db(db_path)

    connection = get_connection(db_path)
    try:
        players = repo.count_players(connection)
    finally:
        connection.close()

    if players == 0:
        from database.seed import export_dataset, seed_database

        seed_database(db_path)
        export_dataset(db_path)

    if not MODEL_PATH.exists():
        from ml.train import train_model

        train_model(db_path, MODEL_PATH)


def _load_model() -> ModelBundle | None:
    """Load the trained model bundle (one model per position group)."""
    if not MODEL_PATH.exists():
        return None
    try:
        return ModelBundle.load(MODEL_PATH)
    except Exception:  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------- helpers ----
def _int_arg(name: str, default: int | None = None) -> int | None:
    raw = request.args.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_arg(name: str, default: float | None = None) -> float | None:
    raw = request.args.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _latest_stats(connection: Any, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Latest season row for each of the given players."""
    result: dict[int, dict[str, Any]] = {}
    for player_id in player_ids:
        rows = repo.get_player_stats(connection, player_id)
        if rows:
            result[player_id] = rows[-1]
    return result


def _team_strength(connection: Any) -> dict[int, float]:
    from ml.features import team_strength_scores

    return team_strength_scores(repo.team_overview(connection))


# ------------------------------------------------------------ application ----
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

    _ensure_ready(DB_PATH)

    _register_routes(app)
    _register_error_handlers(app)

    return app


def _register_routes(app: Flask) -> None:
    # ------------------------------------------------------------- pages ----
    @app.route("/")
    def index():
        connection = get_connection(DB_PATH)
        try:
            overview = repo.overview(connection)
            leaders = repo.league_leaders(connection, limit=8)
            distribution = repo.position_distribution(connection)
            teams = repo.team_overview(connection)
            model_meta = repo.get_model_meta(connection)
            top_players = repo.search_players(connection, limit=5, sort="overall")
        finally:
            connection.close()

        return render_template(
            "index.html",
            overview=overview,
            leaders=leaders,
            distribution=distribution,
            teams=teams,
            model=model_meta,
            top_players=top_players,
            position_labels=POSITION_LABELS,
        )

    @app.route("/players")
    def players():
        query = request.args.get("q", "").strip()
        position = request.args.get("position", "").strip()
        nationality = request.args.get("nationality", "").strip()
        team_id = _int_arg("team")
        age_min = _int_arg("age_min")
        age_max = _int_arg("age_max")
        min_overall = _int_arg("min_overall")
        max_value = _int_arg("max_value")
        sort = request.args.get("sort", "overall")
        order = request.args.get("order", "desc")

        connection = get_connection(DB_PATH)
        try:
            results = repo.search_players(
                connection,
                query=query,
                position=position,
                age_min=age_min,
                age_max=age_max,
                min_overall=min_overall,
                max_value=max_value,
                nationality=nationality,
                team_id=team_id,
                sort=sort,
                order=order,
                limit=60,
            )
            teams = repo.list_teams(connection)
            nationalities = repo.list_nationalities(connection)
            positions = repo.list_positions(connection)
            total = repo.count_players(connection)
        finally:
            connection.close()

        return render_template(
            "players.html",
            results=results,
            teams=teams,
            nationalities=nationalities,
            positions=positions,
            total=total,
            position_labels=POSITION_LABELS,
            filters={
                "q": query,
                "position": position,
                "nationality": nationality,
                "team": team_id or "",
                "age_min": age_min or "",
                "age_max": age_max or "",
                "min_overall": min_overall or "",
                "max_value": max_value or "",
                "sort": sort,
                "order": order,
            },
        )

    @app.route("/players/<int:player_id>")
    def player_detail(player_id: int):
        connection = get_connection(DB_PATH)
        try:
            player = repo.get_player(connection, player_id)
            if not player:
                return render_template("404.html"), 404
            stats = repo.get_player_stats(connection, player_id)
            model = _load_model()
            latest = stats[-1] if stats else None
            team_strength_map = _team_strength(connection)
            strengths = team_strength_map.get(player.get("team_id") or 0, 1.0)
            analysis = score_player(player, player["position"], latest, model, strengths)

            same_position = repo.search_players(connection, position=player["position"], limit=30)
            similar = [
                item.player
                for item in rank_players(
                    same_position,
                    player["position"],
                    stats_by_player=_latest_stats(
                        connection, [row["id"] for row in same_position]
                    ),
                    model=model,
                    limit=4,
                )
                if item.player["id"] != player_id
            ][:3]
        finally:
            connection.close()

        return render_template(
            "player_detail.html",
            player=player,
            stats=stats,
            analysis=analysis,
            similar=similar,
            position_labels=POSITION_LABELS,
        )

    @app.route("/compare")
    def compare():
        raw = request.args.get("ids", "").strip()
        ids = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk.isdigit():
                value = int(chunk)
                if value not in ids:
                    ids.append(value)
        ids = ids[:4]

        connection = get_connection(DB_PATH)
        try:
            selected = repo.get_players(connection, ids) if ids else []
            selected.sort(key=lambda item: ids.index(item["id"]))
            history = repo.get_stats_for_players(connection, [item["id"] for item in selected])
            model = _load_model()
            strength_map = _team_strength(connection)
            analyses = {
                int(item["id"]): score_player(
                    item,
                    item["position"],
                    (_latest_stats(connection, [int(item["id"])]) or {}).get(int(item["id"])),
                    model,
                    strength_map.get(item.get("team_id") or 0, 1.0),
                )
                for item in selected
            }
            all_players = repo.search_players(connection, limit=200, sort="overall")
        finally:
            connection.close()

        return render_template(
            "compare.html",
            selected=selected,
            ids=ids,
            history=history,
            analyses=analyses,
            all_players=all_players,
            position_labels=POSITION_LABELS,
        )

    @app.route("/recommendations")
    def recommendations():
        connection = get_connection(DB_PATH)
        try:
            teams = repo.list_teams(connection)
            team_id = _int_arg("team") or (teams[0]["id"] if teams else None)
            needs = repo.list_team_needs(connection, team_id)
            positions = repo.list_positions(connection)

            selected_position = request.args.get("position", "").strip()
            need = None
            for item in needs:
                if item["position"] == selected_position:
                    need = item
                    break
            if need is None and needs:
                need = needs[0]
                selected_position = need["position"]

            ranked: list[Any] = []
            if need:
                candidates = repo.search_players(connection, limit=200)
                model = _load_model()
                stats_map = _latest_stats(connection, [row["id"] for row in candidates])
                strength_map = _team_strength(connection)
                ranked = rank_players(
                    candidates,
                    need["position"],
                    stats_by_player=stats_map,
                    model=model,
                    team_strength_by_id=strength_map,
                    need=need,
                    limit=8,
                )

            team = repo.get_team(connection, team_id) if team_id else None
        finally:
            connection.close()

        return render_template(
            "recommendations.html",
            teams=teams,
            needs=needs,
            team=team,
            need=need,
            ranked=ranked,
            positions=positions,
            position_labels=POSITION_LABELS,
        )

    @app.route("/assistant")
    def assistant():
        connection = get_connection(DB_PATH)
        try:
            teams = repo.list_teams(connection)
            leaders = repo.league_leaders(connection, limit=3)
        finally:
            connection.close()
        return render_template(
            "assistant.html",
            teams=teams,
            leaders=leaders,
            llm_available=llm_available(),
        )

    # --------------------------------------------------------------- api ----
    @app.route("/api/assistant", methods=["POST"])
    def api_assistant():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Въпросът е празен."}), 400
        connection = get_connection(DB_PATH)
        try:
            answer = ScoutingAssistant(connection).answer(question)
        finally:
            connection.close()
        payload_out = answer.as_dict()
        payload_out["llm_available"] = llm_available()
        return jsonify(payload_out)

    @app.route("/api/players/<int:player_id>/stats")
    def api_player_stats(player_id: int):
        connection = get_connection(DB_PATH)
        try:
            player = repo.get_player(connection, player_id)
            if not player:
                return jsonify({"error": "Играчът не е намерен."}), 404
            stats = repo.get_player_stats(connection, player_id)
        finally:
            connection.close()
        return jsonify({"player": player, "stats": stats})

    @app.route("/api/overview")
    def api_overview():
        connection = get_connection(DB_PATH)
        try:
            data = {
                "overview": repo.overview(connection),
                "leaders": repo.league_leaders(connection, limit=8),
                "distribution": repo.position_distribution(connection),
                "teams": repo.team_overview(connection),
                "model": repo.get_model_meta(connection),
            }
        finally:
            connection.close()
        return jsonify(data)

    @app.route("/health")
    def health():
        connection = get_connection(DB_PATH)
        try:
            payload = {
                "status": "ok",
                "service": "football-intelligence-platform",
                "database": {
                    "players": repo.count_players(connection),
                    "teams": repo.count_teams(connection),
                },
                "model_trained": MODEL_PATH.exists(),
                "llm_configured": llm_available(),
            }
        finally:
            connection.close()
        return jsonify(payload)


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
