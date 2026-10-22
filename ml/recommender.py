"""Recommendation engine.

The final score of a player is a transparent, weighted combination of five
components:

* ``performance``   - probability that the player delivers a strong season
                      (output of the trained logistic regression model);
* ``position_fit``  - how well the attribute profile matches the requested
                      role (position-specific weights);
* ``availability``  - fitness, injury risk and discipline;
* ``value``         - price/quality ratio (value for money);
* ``age_fit``       - age profile that matches the expected career stage.

Every component is also returned separately so the UI can explain *why* a
player was recommended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ml.features import POSITION_GROUP_BY_CODE, player_features, position_group_of
from ml.model import LogisticRegression, ModelBundle

POSITION_LABELS = {
    "GK": "Вратар",
    "CB": "Централен защитник",
    "LB": "Ляв защитник",
    "RB": "Десен защитник",
    "CDM": "Опорен полузащитник",
    "CM": "Централен полузащитник",
    "CAM": "Атакуващ полузащитник",
    "LW": "Ляв крило",
    "RW": "Десен крило",
    "ST": "Нападател",
}

# Позиционни кодове -> групи (GK / DEF / MID / ATT).
POSITION_GROUPS = POSITION_GROUP_BY_CODE

# Which attributes matter for each role and how much (weights sum to 1).
POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"positioning": 0.22, "composure": 0.18, "passing": 0.14, "discipline": 0.12,
           "physical": 0.12, "vision": 0.08, "fitness": 0.08, "stamina": 0.06},
    "CB": {"defending": 0.26, "physical": 0.20, "positioning": 0.14, "composure": 0.12,
           "passing": 0.10, "discipline": 0.08, "teamwork": 0.06, "pace": 0.04},
    "LB": {"pace": 0.24, "defending": 0.20, "stamina": 0.14, "dribbling": 0.12,
           "teamwork": 0.10, "passing": 0.08, "positioning": 0.08, "composure": 0.04},
    "RB": {"pace": 0.24, "defending": 0.20, "stamina": 0.14, "dribbling": 0.12,
           "teamwork": 0.10, "passing": 0.08, "positioning": 0.08, "composure": 0.04},
    "CDM": {"defending": 0.22, "passing": 0.18, "positioning": 0.14, "stamina": 0.14,
            "teamwork": 0.12, "vision": 0.10, "composure": 0.10},
    "CM": {"passing": 0.24, "vision": 0.16, "stamina": 0.14, "teamwork": 0.12,
           "dribbling": 0.10, "composure": 0.10, "positioning": 0.08, "physical": 0.06},
    "CAM": {"vision": 0.24, "passing": 0.18, "dribbling": 0.18, "shooting": 0.12,
            "composure": 0.12, "pace": 0.08, "teamwork": 0.08},
    "LW": {"pace": 0.26, "dribbling": 0.22, "shooting": 0.14, "vision": 0.12,
           "composure": 0.10, "passing": 0.10, "teamwork": 0.06},
    "RW": {"pace": 0.26, "dribbling": 0.22, "shooting": 0.14, "vision": 0.12,
           "composure": 0.10, "passing": 0.10, "teamwork": 0.06},
    "ST": {"shooting": 0.30, "positioning": 0.20, "composure": 0.14, "pace": 0.12,
           "physical": 0.10, "dribbling": 0.08, "vision": 0.06},
}

ATTRIBUTE_LABELS = {
    "pace": "Скорост",
    "shooting": "Удар",
    "passing": "Пасове",
    "dribbling": "Дрибъл",
    "defending": "Защита",
    "physical": "Физика",
    "vision": "Визия",
    "positioning": "Позициониране",
    "composure": "Самообладание",
    "teamwork": "Отборна игра",
    "stamina": "Издръжливост",
    "discipline": "Дисциплина",
    "form": "Форма",
    "fitness": "Физическо състояние",
    "injury_risk": "Риск от нараняване",
}

# Ideal age (and spread) per role.
AGE_PROFILE = {
    "GK": (27.0, 5.0), "CB": (27.0, 4.5), "LB": (25.0, 4.0), "RB": (25.0, 4.0),
    "CDM": (28.0, 4.5), "CM": (26.0, 4.5), "CAM": (25.0, 4.0),
    "LW": (24.0, 4.0), "RW": (24.0, 4.0), "ST": (25.0, 4.5),
}

COMPONENT_WEIGHTS = {
    "performance": 0.34,
    "position_fit": 0.30,
    "availability": 0.14,
    "value": 0.10,
    "age_fit": 0.12,
}

VALUE_REFERENCE = 55_000.0  # EUR per rating point considered a fair price


@dataclass
class Recommendation:
    """A scored player together with the explanation of the score."""

    player: dict[str, Any]
    score: float
    components: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "score": round(self.score, 1),
            "components": {key: round(value, 1) for key, value in self.components.items()},
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


# ------------------------------------------------------------- components ----
def position_fit(player: dict[str, Any], position: str) -> tuple[float, list[tuple[str, float]]]:
    """Weighted attribute match for a role + the strongest matching attributes."""
    weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS["CM"])
    total = 0.0
    contributions: list[tuple[str, float]] = []
    for attribute, weight in weights.items():
        value = float(player.get(attribute, 50))
        total += weight * value
        contributions.append((attribute, value * weight))
    contributions.sort(key=lambda item: item[1], reverse=True)
    return total, contributions


def availability_score(player: dict[str, Any]) -> float:
    """Fitness, injury risk and discipline combined."""
    fitness = float(player.get("fitness", 50))
    injury_risk = float(player.get("injury_risk", 25))
    discipline = float(player.get("discipline", 50))
    stamina = float(player.get("stamina", 50))
    score = 0.42 * fitness + 0.30 * (100 - injury_risk) + 0.16 * discipline + 0.12 * stamina
    return max(0.0, min(100.0, score))


def value_score(player: dict[str, Any]) -> float:
    """Value for money - 100 when the player costs less than the reference.

    The curve is compressed (square root) so that a price difference does not
    dominate the final ranking.
    """
    overall = max(1, int(player.get("overall", 1)))
    market_value = float(player.get("market_value", 0))
    ratio = market_value / overall
    if ratio <= 0:
        return 100.0
    if ratio <= VALUE_REFERENCE:
        return 100.0
    return max(20.0, min(100.0, 100.0 * math.sqrt(VALUE_REFERENCE / ratio)))


def age_fit_score(player: dict[str, Any], position: str) -> float:
    ideal, sigma = AGE_PROFILE.get(position, AGE_PROFILE["CM"])
    age = float(player.get("age", 25))
    score = 100.0 * math.exp(-((age - ideal) ** 2) / (2 * sigma ** 2))
    if age <= 21:
        score = min(100.0, score + 8.0)  # extra headroom for a young player
    return max(0.0, min(100.0, score))


def _fallback_logit(player: dict[str, Any], stats: dict[str, Any] | None) -> float:
    """Heuristic used when the model has not been trained yet."""
    base = float(player.get("overall", 50)) - 60.0
    contribution = 0.0
    if stats and stats.get("minutes"):
        contribution = (
            float(stats.get("goals", 0)) + float(stats.get("assists", 0))
        ) / float(stats["minutes"]) * 90
    return base + min(contribution, 0.6) * 4.0


def performance_logit(
    player: dict[str, Any],
    stats: dict[str, Any] | None,
    model: LogisticRegression | ModelBundle | None,
    team_strength: float = 1.0,
) -> float:
    """Raw model output (logit) for a player - used for ranking.

    With a :class:`ModelBundle` the model of the player's own position group is
    used, so an attacker is never compared with a goalkeeper by raw numbers.
    """
    if isinstance(model, ModelBundle):
        if model.models:
            return model.logit(player, stats, team_strength)
        return _fallback_logit(player, stats)

    features = player_features(player, stats, team_strength)
    if model is None or not model.weights:
        return _fallback_logit(player, stats)
    return model.decision_function([features])[0]


def performance_score(
    player: dict[str, Any],
    stats: dict[str, Any] | None,
    model: LogisticRegression | ModelBundle | None,
    team_strength: float = 1.0,
) -> float:
    """Probability of a strong season, converted to a 0-100 score."""
    if not isinstance(model, ModelBundle) and (model is None or not model.weights):
        base = float(player.get("overall", 50))
        contribution = 0.0
        if stats and stats.get("minutes"):
            contribution = (
                float(stats.get("goals", 0)) + float(stats.get("assists", 0))
            ) / float(stats["minutes"]) * 90
        return max(0.0, min(100.0, base * 0.85 + min(contribution / 0.55, 1.5) * 10.0))

    logit = performance_logit(player, stats, model, team_strength)
    # Softened sigmoid - the plain probability saturates around 0.99.
    return max(0.0, min(100.0, 100.0 / (1.0 + math.exp(-logit / 2.5))))


# ------------------------------------------------------------ explanation ----
def _format_money(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн. €"
    if value >= 1_000:
        return f"{value / 1_000:.0f} хил. €"
    return f"{value:.0f} €"


def build_reasons(
    player: dict[str, Any],
    stats: dict[str, Any] | None,
    components: dict[str, float],
    contributions: Sequence[tuple[str, float]],
    position: str,
) -> tuple[list[str], list[str]]:
    """Human readable justification of the recommendation (in Bulgarian)."""
    reasons: list[str] = []
    warnings: list[str] = []

    position_name = POSITION_LABELS.get(position, position)

    # 1. Attribute strengths for the requested role.
    for attribute, _value in contributions[:3]:
        raw = float(player.get(attribute, 50))
        if raw >= 60:
            reasons.append(
                f"Силна {ATTRIBUTE_LABELS.get(attribute, attribute).lower()} ({int(raw)}/99) "
                f"за ролята „{position_name}“"
            )

    # 2. Model performance probability.
    if components["performance"] >= 60:
        reasons.append(
            f"Моделът предвижда силни изненади в следващия сезон "
            f"({components['performance']:.0f}% вероятност за силна форма)"
        )
    elif components["performance"] < 35:
        warnings.append("Моделът не очаква силна следваща кампания")

    # 3. Availability.
    injury_risk = float(player.get("injury_risk", 25))
    if injury_risk <= 15:
        reasons.append(f"Нисък риск от нараняване ({int(injury_risk)}%) - надеждна ротация")
    elif injury_risk >= 30:
        warnings.append(f"Повишен риск от нараняване ({int(injury_risk)}%)")

    # 4. Form.
    form = float(player.get("form", 50))
    if form >= 78:
        reasons.append(f"Отлична текуща форма ({int(form)}/99)")
    elif form <= 45:
        warnings.append(f"Занижена форма в момента ({int(form)}/99)")

    # 5. Potential headroom.
    gap = int(player.get("potential", 0)) - int(player.get("overall", 0))
    if gap >= 8:
        reasons.append(
            f"Голям потенциал за развитие: {int(player.get('overall'))} → "
            f"{int(player.get('potential'))} (разгъване +{gap})"
        )

    # 6. Value for money.
    if components["value"] >= 80:
        reasons.append(
            f"Изгодна сделка - {_format_money(float(player.get('market_value', 0)))} "
            f"за оценка {int(player.get('overall', 0))}"
        )
    elif components["value"] <= 35:
        warnings.append(
            f"Висока цена за класацията ({_format_money(float(player.get('market_value', 0)))})"
        )

    # 7. Production from the data.
    if stats and stats.get("minutes"):
        minutes = float(stats["minutes"])
        contribution = (float(stats.get("goals", 0)) + float(stats.get("assists", 0))) / minutes * 90
        if contribution >= 0.25:
            reasons.append(
                f"Ефективност в последния сезон: {contribution:.2f} гола+асиста на 90 минути "
                f"({int(stats.get('goals', 0))}+{int(stats.get('assists', 0))} в "
                f"{int(stats.get('appearances', 0))} мача)"
            )
        pass_accuracy = float(stats.get("pass_accuracy", 0) or 0)
        if pass_accuracy >= 80:
            reasons.append(f"Висока точност на пасовете в лигата ({pass_accuracy:.0f}%)")

    # 8. Age context.
    age = int(player.get("age", 0))
    ideal = AGE_PROFILE.get(position, (26.0, 5.0))[0]
    if age <= 21:
        reasons.append(f"Млад играч ({age} г.) с висок таван на развитие")
    elif age > ideal + 7:
        warnings.append(f"Възрастта ({age} г.) е над оптималната за тази позиция")

    if not reasons:
        reasons.append("Сбалансиран профил, който отговаря на изискванията на отбора")

    return reasons, warnings


# --------------------------------------------------------------- ranking ----
def score_player(
    player: dict[str, Any],
    position: str,
    stats: dict[str, Any] | None = None,
    model: LogisticRegression | ModelBundle | None = None,
    team_strength: float = 1.0,
    need: dict[str, Any] | None = None,
) -> Recommendation:
    """Score a single player for a requested role."""
    fit, contributions = position_fit(player, position)
    components = {
        "performance": performance_score(player, stats, model, team_strength),
        "position_fit": fit,
        "availability": availability_score(player),
        "value": value_score(player),
        "age_fit": age_fit_score(player, position),
    }
    score = sum(components[key] * weight for key, weight in COMPONENT_WEIGHTS.items())

    # Hard filters from the club request reduce the score instead of hiding
    # the player, so the user can still see "why not".
    if need:
        min_overall = int(need.get("min_overall") or 0)
        if min_overall and int(player.get("overall", 0)) < min_overall:
            score -= 12.0
        max_age = int(need.get("max_age") or 0)
        if max_age and int(player.get("age", 0)) > max_age:
            score -= 8.0
        max_value = int(need.get("max_value") or 0)
        if max_value and int(player.get("market_value", 0)) > max_value:
            score -= 6.0

    reasons, warnings = build_reasons(
        player, stats, components, contributions, position
    )
    if need:
        notes = need.get("notes")
        if notes:
            reasons.insert(0, f"Отговаря на заявката: {notes}")

    return Recommendation(
        player=player,
        score=max(0.0, min(100.0, score)),
        components=components,
        reasons=reasons[:5],
        warnings=warnings[:3],
    )


def _percentile_within_group(
    players: Sequence[dict[str, Any]],
    raw_scores: dict[int, float],
) -> dict[int, float]:
    """Convert raw model outputs into percentiles inside each position group.

    A striker and a centre-back are not comparable in raw terms (a striker
    scores, a centre-back does not). Percentiles make the performance signal
    relative to the player's own peers, which keeps the recommendation fair.
    """
    groups: dict[str, list[int]] = {}
    for player in players:
        groups.setdefault(POSITION_GROUPS.get(player.get("position", ""), "OTHER"), []).append(
            int(player["id"])
        )

    percentiles: dict[int, float] = {}
    for _group, ids in groups.items():
        ordered = sorted(ids, key=lambda pid: raw_scores.get(pid, 0.0))
        size = len(ordered)
        for index, player_id in enumerate(ordered):
            percentiles[player_id] = 100.0 * (index + 1) / size if size else 50.0
    return percentiles


def rank_players(
    players: Iterable[dict[str, Any]],
    position: str,
    stats_by_player: dict[int, dict[str, Any]] | None = None,
    model: LogisticRegression | ModelBundle | None = None,
    team_strength_by_id: dict[int, float] | None = None,
    need: dict[str, Any] | None = None,
    limit: int = 10,
    natural_positions_only: bool = False,
) -> list[Recommendation]:
    """Rank a set of players for a role, best first."""
    stats_by_player = stats_by_player or {}
    team_strength_by_id = team_strength_by_id or {}

    candidates = [
        player for player in players
        if not natural_positions_only or player.get("position") == position
    ]
    if not candidates:
        return []

    raw_performance: dict[int, float] = {}
    for player in candidates:
        stats = stats_by_player.get(int(player["id"]))
        strength = team_strength_by_id.get(int(player.get("team_id") or 0), 1.0)
        raw_performance[int(player["id"])] = performance_logit(player, stats, model, strength)

    percentiles = _percentile_within_group(candidates, raw_performance)

    recommendations: list[Recommendation] = []
    for player in candidates:
        stats = stats_by_player.get(int(player["id"]))
        strength = team_strength_by_id.get(int(player.get("team_id") or 0), 1.0)
        recommendation = score_player(player, position, stats, model, strength, need)

        # The model signal is combined with the absolute quality of the player
        # so that a strong player is never ranked below a weaker one just
        # because of a different position group.
        percentile = percentiles.get(int(player["id"]), 50.0)
        quality = float(player.get("overall", 50))
        recommendation.components["performance"] = 0.5 * percentile + 0.5 * quality

        recommendation.score = max(
            0.0,
            min(
                100.0,
                sum(
                    recommendation.components[key] * weight
                    for key, weight in COMPONENT_WEIGHTS.items()
                ),
            ),
        )
        recommendations.append(recommendation)

    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:limit]
