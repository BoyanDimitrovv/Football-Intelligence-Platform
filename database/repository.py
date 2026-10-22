"""Data access layer.

All SQL used by the application lives here so that routes and templates stay
free of SQL and the data model can evolve in a single place.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Sequence

from database.db import get_connection, query_all, query_one, scalar

POSITION_LABELS = {
    "GK": "Вратар",
    "CB": "Централен защитник",
    "LB": "Ляв защитник",
    "RB": "Десен защитник",
    "CDM": "Полузащитник (опорен)",
    "CM": "Полузащитник (централен)",
    "CAM": "Атакуващ полузащитник",
    "LW": "Ляв крило",
    "RW": "Десен крило",
    "ST": "Нападател",
}

POSITION_GROUPS = {
    "Вратар": ["GK"],
    "Защитник": ["CB", "LB", "RB"],
    "Полузащитник": ["CDM", "CM", "CAM"],
    "Нападател": ["LW", "RW", "ST"],
}

SORTS = {
    "overall": "p.overall",
    "potential": "p.potential",
    "age": "p.age",
    "value": "p.market_value",
    "name": "p.name",
    "form": "p.form",
}


# ------------------------------------------------------------------ teams ----
def list_teams(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return query_all(
        connection,
        """SELECT t.*, COUNT(p.id) AS player_count
           FROM teams t LEFT JOIN players p ON p.team_id = t.id
           GROUP BY t.id ORDER BY t.name""",
    )


def get_team(connection: sqlite3.Connection, team_id: int) -> dict[str, Any] | None:
    return query_one(connection, "SELECT * FROM teams WHERE id = ?", (team_id,))


def list_team_needs(connection: sqlite3.Connection, team_id: int | None = None) -> list[dict[str, Any]]:
    if team_id is None:
        return query_all(
            connection,
            """SELECT n.*, t.name AS team_name, t.short_name AS team_short, t.color AS team_color
               FROM team_needs n JOIN teams t ON t.id = n.team_id
               ORDER BY t.name, n.priority""",
        )
    return query_all(
        connection,
        """SELECT n.*, t.name AS team_name, t.short_name AS team_short, t.color AS team_color
           FROM team_needs n JOIN teams t ON t.id = n.team_id
           WHERE n.team_id = ? ORDER BY n.priority""",
        (team_id,),
    )


# ---------------------------------------------------------------- players ----
PLAYER_SELECT = """
    SELECT p.*, t.name AS team_name, t.short_name AS team_short, t.color AS team_color,
           t.league AS team_league
    FROM players p LEFT JOIN teams t ON t.id = p.team_id
"""


def get_player(connection: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    return query_one(connection, PLAYER_SELECT + " WHERE p.id = ?", (player_id,))


def get_players(connection: sqlite3.Connection, player_ids: Sequence[int]) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    placeholders = ", ".join("?" for _ in player_ids)
    return query_all(
        connection,
        PLAYER_SELECT + f" WHERE p.id IN ({placeholders})",
        tuple(player_ids),
    )


def find_players_by_name(connection: sqlite3.Connection, name: str, limit: int = 5) -> list[dict[str, Any]]:
    like = f"%{name.strip()}%"
    return query_all(
        connection,
        PLAYER_SELECT + " WHERE p.name LIKE ? ORDER BY p.overall DESC LIMIT ?",
        (like, limit),
    )


def search_players(
    connection: sqlite3.Connection,
    *,
    query: str = "",
    position: str | list[str] = "",
    age_min: int | None = None,
    age_max: int | None = None,
    min_overall: int | None = None,
    max_value: int | None = None,
    nationality: str = "",
    team_id: int | None = None,
    sort: str = "overall",
    order: str = "desc",
    limit: int = 60,
) -> list[dict[str, Any]]:
    """Search and filter players with an optional free-text query.

    `position` accepts a single code or a list of codes (position groups).
    """
    where: list[str] = []
    params: list[Any] = []

    if query:
        where.append("(p.name LIKE ? OR p.nationality LIKE ? OR t.name LIKE ?)")
        like = f"%{query.strip()}%"
        params.extend([like, like, like])
    if position:
        positions = [position] if isinstance(position, str) else list(position)
        where.append(f"p.position IN ({', '.join('?' for _ in positions)})")
        params.extend(positions)
    if age_min is not None:
        where.append("p.age >= ?")
        params.append(age_min)
    if age_max is not None:
        where.append("p.age <= ?")
        params.append(age_max)
    if min_overall is not None:
        where.append("p.overall >= ?")
        params.append(min_overall)
    if max_value is not None:
        where.append("p.market_value <= ?")
        params.append(max_value)
    if nationality:
        where.append("p.nationality = ?")
        params.append(nationality)
    if team_id is not None:
        where.append("p.team_id = ?")
        params.append(team_id)

    clause = f" WHERE {' AND '.join(where)}" if where else ""
    sort_column = SORTS.get(sort, "p.overall")
    direction = "ASC" if str(order).lower() == "asc" else "DESC"
    params.append(limit)

    return query_all(
        connection,
        PLAYER_SELECT + clause + f" ORDER BY {sort_column} {direction}, p.name ASC LIMIT ?",
        params,
    )


def count_players(connection: sqlite3.Connection) -> int:
    return int(scalar(connection, "SELECT COUNT(*) FROM players"))


def count_teams(connection: sqlite3.Connection) -> int:
    return int(scalar(connection, "SELECT COUNT(*) FROM teams"))


def list_nationalities(connection: sqlite3.Connection) -> list[str]:
    rows = query_all(
        connection,
        "SELECT DISTINCT nationality FROM players ORDER BY nationality",
    )
    return [row["nationality"] for row in rows]


def list_positions(connection: sqlite3.Connection) -> list[dict[str, str]]:
    rows = query_all(
        connection,
        "SELECT DISTINCT position FROM players ORDER BY position",
    )
    return [
        {"code": row["position"], "label": POSITION_LABELS.get(row["position"], row["position"])}
        for row in rows
    ]


# ------------------------------------------------------------------ stats ----
def get_player_stats(connection: sqlite3.Connection, player_id: int) -> list[dict[str, Any]]:
    return query_all(
        connection,
        "SELECT * FROM player_stats WHERE player_id = ? ORDER BY season",
        (player_id,),
    )


def get_stats_for_players(
    connection: sqlite3.Connection, player_ids: Sequence[int]
) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    placeholders = ", ".join("?" for _ in player_ids)
    return query_all(
        connection,
        f"""SELECT s.*, p.name AS player_name, p.position
            FROM player_stats s JOIN players p ON p.id = s.player_id
            WHERE s.player_id IN ({placeholders}) ORDER BY s.season""",
        tuple(player_ids),
    )


def latest_season(connection: sqlite3.Connection) -> str:
    return str(scalar(connection, "SELECT MAX(season) FROM player_stats", default="2024/25"))


def league_leaders(connection: sqlite3.Connection, limit: int = 8) -> list[dict[str, Any]]:
    season = latest_season(connection)
    return query_all(
        connection,
        """SELECT p.id, p.name, p.position, p.overall, t.name AS team_name, t.color AS team_color,
                  s.goals, s.assists, s.appearances, s.minutes, s.pass_accuracy, s.shots_on_target
           FROM player_stats s
           JOIN players p ON p.id = s.player_id
           LEFT JOIN teams t ON t.id = p.team_id
           WHERE s.season = ?
           ORDER BY (s.goals + s.assists) DESC, s.goals DESC LIMIT ?""",
        (season, limit),
    )


def position_distribution(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return query_all(
        connection,
        """SELECT p.position, COUNT(*) AS count FROM players p GROUP BY p.position ORDER BY count DESC""",
    )


def team_composition(connection: sqlite3.Connection, team_id: int | None = None) -> list[dict[str, Any]]:
    if team_id is None:
        return query_all(
            connection,
            """SELECT p.position, COUNT(*) AS count, ROUND(AVG(p.overall), 1) AS avg_overall
               FROM players p GROUP BY p.position ORDER BY count DESC""",
        )
    return query_all(
        connection,
        """SELECT p.position, COUNT(*) AS count, ROUND(AVG(p.overall), 1) AS avg_overall
           FROM players p WHERE p.team_id = ? GROUP BY p.position ORDER BY count DESC""",
        (team_id,),
    )


def team_overview(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Per-team summary used on the dashboard."""
    season = latest_season(connection)
    return query_all(
        connection,
        """SELECT t.id, t.name, t.short_name, t.color, t.league,
                  COUNT(p.id) AS squad_size,
                  ROUND(AVG(p.overall), 1) AS avg_overall,
                  ROUND(AVG(p.age), 1) AS avg_age,
                  SUM(COALESCE(s.goals, 0)) AS goals,
                  SUM(COALESCE(s.assists, 0)) AS assists
           FROM teams t
           LEFT JOIN players p ON p.team_id = t.id
           LEFT JOIN player_stats s ON s.player_id = p.id AND s.season = ?
           GROUP BY t.id ORDER BY avg_overall DESC""",
        (season,),
    )


# ----------------------------------------------------------- model / meta ----
def save_model_meta(
    connection: sqlite3.Connection,
    name: str,
    trained_at: str,
    metrics: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    connection.execute(
        """INSERT INTO model_meta (name, trained_at, metrics, weights) VALUES (?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET trained_at = excluded.trained_at,
               metrics = excluded.metrics, weights = excluded.weights""",
        (name, trained_at, json.dumps(metrics), json.dumps(weights)),
    )
    connection.commit()


def get_model_meta(connection: sqlite3.Connection, name: str = "scouting-logistic") -> dict[str, Any] | None:
    row = query_one(connection, "SELECT * FROM model_meta WHERE name = ?", (name,))
    if not row:
        return None
    row["metrics"] = json.loads(row["metrics"])
    row["weights"] = json.loads(row["weights"])
    return row


# ------------------------------------------------------------- convenience ----
def overview(connection: sqlite3.Connection) -> dict[str, Any]:
    """Aggregated numbers shown on the dashboard."""
    season = latest_season(connection)
    return {
        "players": count_players(connection),
        "teams": int(scalar(connection, "SELECT COUNT(*) FROM teams")),
        "seasons": int(scalar(connection, "SELECT COUNT(DISTINCT season) FROM player_stats")),
        "stat_rows": int(scalar(connection, "SELECT COUNT(*) FROM player_stats")),
        "latest_season": season,
        "goals": int(scalar(connection, "SELECT COALESCE(SUM(goals), 0) FROM player_stats")),
        "assists": int(scalar(connection, "SELECT COALESCE(SUM(assists), 0) FROM player_stats")),
        "avg_age": round(float(scalar(connection, "SELECT AVG(age) FROM players", default=0)), 1),
        "total_value": int(scalar(connection, "SELECT COALESCE(SUM(market_value), 0) FROM players")),
    }
