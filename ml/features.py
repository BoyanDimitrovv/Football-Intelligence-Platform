"""Feature engineering for the scouting model.

Every player-season observation is turned into a fixed list of numeric
features. The same builder is used during training and at prediction time so
the model never sees a different representation.
"""

from __future__ import annotations

from typing import Any, Sequence

FEATURE_NAMES = (
    "overall",
    "potential_gap",
    "age",
    "form",
    "fitness",
    "discipline",
    "durability",
    "goal_contribution",
    "pass_accuracy",
    "duels_won",
    "shot_accuracy",
    "clean_sheets_per_game",
    "defensive_actions",
    "team_strength",
)

# Позиционни кодове -> позиционни групи. Сравнява се само вътре в група,
# защото големите нападатели и чистите защитници нямат еднакъв профил.
POSITION_GROUP_BY_CODE = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID",
    "LW": "ATT", "RW": "ATT", "ST": "ATT",
}

POSITION_GROUP_LABELS = {
    "GK": "Вратари",
    "DEF": "Защитници",
    "MID": "Полузащитници",
    "ATT": "Нападатели",
}


def position_group_of(player: dict[str, Any]) -> str:
    """Позиционната група (GK/DEF/MID/ATT) на играча."""
    return POSITION_GROUP_BY_CODE.get(str(player.get("position", "")).upper(), "MID")


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def player_features(
    player: dict[str, Any],
    stats: dict[str, Any] | None = None,
    team_strength: float = 1.0,
) -> list[float]:
    """Build the feature vector of a single player (+ optional latest season)."""
    stats = stats or {}
    appearances = float(stats.get("appearances", 0) or 0)
    minutes = float(stats.get("minutes", 0) or 0)
    goals = float(stats.get("goals", 0) or 0)
    assists = float(stats.get("assists", 0) or 0)
    shots = float(stats.get("shots", 0) or 0)
    shots_on_target = float(stats.get("shots_on_target", 0) or 0)

    goal_contribution = _safe_div(goals + assists, minutes) * 90.0
    shot_accuracy = _safe_div(shots_on_target, shots) * 100.0
    clean_sheets_per_game = _safe_div(float(stats.get("clean_sheets", 0) or 0), appearances)
    defensive_actions = _safe_div(
        float(stats.get("tackles", 0) or 0) + float(stats.get("interceptions", 0) or 0),
        minutes,
    ) * 90.0

    return [
        float(player.get("overall", 0)),
        float(player.get("potential", 0) - player.get("overall", 0)),
        float(player.get("age", 25)),
        float(player.get("form", 50)),
        float(player.get("fitness", 50)),
        float(player.get("discipline", 50)),
        _safe_div(appearances, 34.0),
        goal_contribution,
        float(stats.get("pass_accuracy", 0) or 0),
        float(stats.get("duels_won_pct", 0) or 0),
        shot_accuracy,
        clean_sheets_per_game,
        defensive_actions,
        team_strength,
    ]


def performance_index(stats: dict[str, Any], position_group: str = "MID") -> float:
    """Композитен индекс 0-100 за един сезон (използва се за етикетите).

    Индексът е позиционно- aware: при вратар се броятят чисти листове и
    точността на пасовете, при защитник - дуелите и подчистяванията, при
    полузащитник - пасовете и приноса, при нападател - головете и точността
    на ударите. Така един и същ индекс означава еднакво „изпълнение" за
    играчи с различни функции.
    """
    minutes = float(stats.get("minutes", 0) or 0)
    goals = float(stats.get("goals", 0) or 0)
    assists = float(stats.get("assists", 0) or 0)
    appearances = float(stats.get("appearances", 0) or 0)
    pass_accuracy = float(stats.get("pass_accuracy", 0) or 0)
    duels_won = float(stats.get("duels_won_pct", 0) or 0)
    tackles = float(stats.get("tackles", 0) or 0)
    interceptions = float(stats.get("interceptions", 0) or 0)
    clean_sheets = float(stats.get("clean_sheets", 0) or 0)
    shots = float(stats.get("shots", 0) or 0)
    shots_on_target = float(stats.get("shots_on_target", 0) or 0)

    ninetys = minutes / 90.0
    contribution = _safe_div(goals + assists, ninetys)
    actions_per_90 = _safe_div(tackles + interceptions, ninetys)
    clean_ratio = _safe_div(clean_sheets, appearances)
    shot_accuracy = _safe_div(shots_on_target, shots)
    availability = min(appearances / 32.0, 1.0)

    if position_group == "GK":
        score = min(clean_ratio / 0.40, 1.0) * 46.0
        score += min(pass_accuracy / 92.0, 1.0) * 24.0
        score += min(duels_won / 62.0, 1.0) * 10.0
        score += availability * 20.0
    elif position_group == "DEF":
        score = min(actions_per_90 / 4.2, 1.0) * 26.0
        score += min(duels_won / 62.0, 1.0) * 24.0
        score += min(pass_accuracy / 90.0, 1.0) * 18.0
        score += min(clean_ratio / 0.12, 1.0) * 12.0
        score += availability * 20.0
    elif position_group == "MID":
        score = min(contribution / 0.32, 1.0) * 36.0
        score += min(pass_accuracy / 90.0, 1.0) * 26.0
        score += min(duels_won / 60.0, 1.0) * 16.0
        score += availability * 22.0
    else:  # ATT
        score = min(contribution / 0.62, 1.0) * 44.0
        score += min(shot_accuracy / 0.55, 1.0) * 18.0
        score += min(pass_accuracy / 85.0, 1.0) * 14.0
        score += availability * 24.0

    return round(min(score, 100.0), 2)


def _normalized_indices(
    rows: Sequence[dict[str, Any]],
    indices: Sequence[float],
) -> list[float]:
    """Normalise the performance index against the player's *natural* position.

    A striker and a winger do not have the same output, so their raw indices
    are not comparable. After the normalisation every player is measured
    against his own peers and the model learns quality instead of position.
    """
    buckets: dict[str, list[float]] = {}
    for row, value in zip(rows, indices):
        buckets.setdefault(str(row["player"].get("position", "")), []).append(value)

    stats: dict[str, tuple[float, float]] = {}
    for position, values in buckets.items():
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        stats[position] = (mean, variance ** 0.5)

    normalized: list[float] = []
    for row, value in zip(rows, indices):
        mean, deviation = stats.get(str(row["player"].get("position", "")), (0.0, 0.0))
        if deviation < 1e-6:
            normalized.append(0.0)
        else:
            normalized.append((value - mean) / deviation)
    return normalized


def build_training_rows(
    players: Sequence[dict[str, Any]],
    stats_by_player: dict[int, list[dict[str, Any]]],
    team_strength_by_id: dict[int, float] | None = None,
) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    """Create (X, y, rows) for the whole dataset.

    The label marks the top 40% of observations **inside each position group**
    by the position-aware performance index, i.e. the seasons in which the
    player clearly delivered *compared to his own peers*. Without this the
    model would simply learn "be an attacker", because attackers produce more
    goals+assists than everybody else.
    """
    team_strength_by_id = team_strength_by_id or {}
    rows: list[dict[str, Any]] = []
    X: list[list[float]] = []
    y: list[int] = []

    for player in players:
        team_strength = team_strength_by_id.get(player.get("team_id") or 0, 1.0)
        group = position_group_of(player)
        for stats in stats_by_player.get(player["id"], []):
            rows.append({"player": player, "stats": stats, "group": group})
            X.append(player_features(player, stats, team_strength))

    if not rows:
        return X, y, rows

    indices = [performance_index(row["stats"], row["group"]) for row in rows]
    normalized = _normalized_indices(rows, indices)

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["group"], []).append(index)

    for _group, group_indices in groups.items():
        ordered = sorted(normalized[index] for index in group_indices)
        cutoff_index = int(len(ordered) * 0.6)
        cutoff = ordered[cutoff_index] if cutoff_index < len(ordered) else ordered[-1]
        for index in group_indices:
            y.append(1 if normalized[index] >= cutoff else 0)

    return X, y, rows


def team_strength_scores(teams: Sequence[dict[str, Any]]) -> dict[int, float]:
    """Normalize the average squad quality per team into a 0.8 - 1.2 factor."""
    if not teams:
        return {}
    averages = [float(team.get("avg_overall") or 0) for team in teams]
    if not averages:
        return {}
    highest = max(averages) or 1.0
    lowest = min(averages)
    spread = (highest - lowest) or 1.0
    result: dict[int, float] = {}
    for team in teams:
        normalized = (float(team.get("avg_overall") or 0) - lowest) / spread
        result[int(team["id"])] = round(0.85 + normalized * 0.35, 4)
    return result
