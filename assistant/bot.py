"""AI assistant for the platform.

The assistant works fully offline: incoming questions are classified into
intents, the relevant entities (players, teams, positions, filters) are
extracted and answered from the database.

If an LLM API key is present in the environment (`OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`) the module exposes `llm_available()` so the application
can transparently enrich an answer with an external model - the local engine
stays as the fallback so the demo never breaks without internet access.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from database.repository import (
    POSITION_LABELS,
    find_players_by_name,
    get_player,
    get_player_stats,
    get_players,
    list_teams,
    search_players,
)
from ml.recommender import POSITION_LABELS as ML_POSITION_LABELS

POSITION_LABELS = POSITION_LABELS or ML_POSITION_LABELS

# keyword -> (role used for the attribute weights, positions that may fill the role)
POSITION_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "вратар": ("GK", ["GK"]),
    "вратари": ("GK", ["GK"]),
    "защитник": ("CB", ["CB", "LB", "RB"]),
    "защитници": ("CB", ["CB", "LB", "RB"]),
    "централен защитник": ("CB", ["CB"]),
    "централни защитници": ("CB", ["CB"]),
    "лев защитник": ("LB", ["LB"]),
    "ляво крило": ("LW", ["LW"]),
    "дясна отбрана": ("RB", ["RB"]),
    "десен защитник": ("RB", ["RB"]),
    "полузащитник": ("CM", ["CDM", "CM", "CAM"]),
    "полузащитници": ("CM", ["CDM", "CM", "CAM"]),
    "опорен полузащитник": ("CDM", ["CDM"]),
    "централен полузащитник": ("CM", ["CM"]),
    "атакуващ полузащитник": ("CAM", ["CAM"]),
    "крило": ("LW", ["LW", "RW"]),
    "крила": ("LW", ["LW", "RW"]),
    "дясна крила": ("RW", ["RW"]),
    "нападател": ("ST", ["ST", "LW", "RW"]),
    "нападатели": ("ST", ["ST", "LW", "RW"]),
    "форвард": ("ST", ["ST"]),
    "център-нападател": ("ST", ["ST"]),
}

GREETING_WORDS = ("здравей", "здрасти", "хай", "привет", "hello", "hi ", "добър ден", "добро утро")
HELP_WORDS = ("помощ", "какво можеш", "какво умееш", "какво знаеш", "команди")
COMPARE_WORDS = ("сравни", "сравнение", "разлика", "кой е по-добър", "кой е по-силен")
RECOMMEND_WORDS = ("препоръчай", "препоръча", "търси футболист", "търси играч", "кого да вземем",
                   "подходящ", "кого да подпишем", "нуждая се", "търси ми")
TOP_WORDS = ("топ", "най-добър", "най-добри", "най-слаби", "лидер", "водещ", "класация",
             "статистика", "статистики", "рекорд")
LIST_WORDS = ("покажи", "изведи", "списък", "всички", "кой играчи", "кой футболисти", "избери")
INFO_WORDS = ("кой е", "разкажи", "информация", "профил", "покажи ми", "данни за")
VALUE_WORDS = ("цена", "стойност", "пазарна", "трансфер", "колко струва", "заплата", "скъпи")
AGE_WORDS = ("млад", "млади", "талант", "потенциал", "под 23", "до 23", "под 25", "до 25")
FITNESS_WORDS = ("здрав", "физическо", "нараняване", "травма", "издръжлив")


@dataclass
class AssistantAnswer:
    """Structured answer that the UI can render in a richer way than plain text."""

    reply: str
    intent: str = "text"
    cards: list[dict[str, Any]] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "intent": self.intent,
            "cards": self.cards,
            "comparison": self.comparison,
            "suggestions": self.suggestions,
            "meta": self.meta,
        }


def llm_available() -> bool:
    """True when an external LLM key is configured (optional enhancement)."""
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_positions(text: str) -> tuple[str, list[str]]:
    """Detect the requested role and the positions that may fill it.

    Returns ``(role, candidates)`` where `role` is the position whose
    attribute weights are used and `candidates` are all positions that can
    reasonably cover that role (e.g. "защитник" -> CB, LB, RB).
    """
    role = ""
    candidates: list[str] = []
    working = text
    for keyword in sorted(POSITION_KEYWORDS, key=len, reverse=True):
        if keyword not in working:
            continue
        # Consume the match so that shorter keywords do not match inside it
        # ("полузащитник" must not be read as "защитник").
        working = working.replace(keyword, " " * len(keyword))
        keyword_role, keyword_candidates = POSITION_KEYWORDS[keyword]
        if not role:
            role = keyword_role
        for code in keyword_candidates:
            if code not in candidates:
                candidates.append(code)
    return role, candidates


def _extract_number(text: str, patterns: Sequence[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_age_filters(text: str) -> tuple[int | None, int | None]:
    age_min = _extract_number(text, (r"под\s*(\d{2})\s*(?:години|г\.|годишен)", r"до\s*(\d{2})\s*(?:години|г\.)"))
    age_max = _extract_number(text, (r"(\d{2})\s*(?:-|–)\s*\d{2}\s*години",))
    if re.search(r"под\s*23", text):
        age_max = 23
    if re.search(r"под\s*25", text):
        age_max = 25
    if re.search(r"над\s*(\d{2})", text):
        age_min = int(re.search(r"над\s*(\d{2})", text).group(1))
    return age_min, age_max


def _player_summary(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": player["id"],
        "name": player["name"],
        "position": player["position"],
        "position_label": POSITION_LABELS.get(player["position"], player["position"]),
        "team": player.get("team_name") or "Без отбор",
        "team_color": player.get("team_color") or "#22c55e",
        "age": player["age"],
        "nationality": player["nationality"],
        "overall": player["overall"],
        "potential": player["potential"],
        "market_value": player["market_value"],
        "form": player.get("form", 0),
        "url": f"/players/{player['id']}",
    }


class ScoutingAssistant:
    """Rule based assistant with entity extraction over the football database."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.teams = list_teams(connection)
        self.players = search_players(connection, limit=500)

    # ------------------------------------------------------------ helpers ----
    def _find_teams(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for team in self.teams:
            short = str(team["short_name"]).lower()
            name = str(team["name"]).lower()
            if short in text or name in text:
                matches.append(team)
        return matches

    def _find_players(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for player in self.players:
            name = str(player["name"]).lower()
            if name in text:
                matches.append(player)
                continue
            parts = name.split()
            if len(parts) >= 2 and parts[-1] in text and len(parts[-1]) > 4:
                matches.append(player)
        return matches

    def _stats_of(self, player_id: int) -> dict[str, Any] | None:
        rows = get_player_stats(self.connection, player_id)
        return rows[-1] if rows else None

    # ------------------------------------------------------------- intents ----
    def answer(self, question: str) -> AssistantAnswer:
        text = _normalize(question)
        if not text:
            return self._greeting()

        if any(word in text for word in HELP_WORDS):
            return self._help()

        mentioned_players = self._find_players(text)
        mentioned_teams = self._find_teams(text)

        if any(word in text for word in COMPARE_WORDS) or len(mentioned_players) >= 2:
            return self._compare(mentioned_players)

        if any(word in text for word in RECOMMEND_WORDS):
            return self._recommend(text, mentioned_teams)

        if any(word in text for word in INFO_WORDS) and mentioned_players:
            return self._player_info(mentioned_players[0])

        if ("статистик" in text or "профил" in text) and mentioned_players:
            return self._player_info(mentioned_players[0])

        if any(word in text for word in VALUE_WORDS) and mentioned_players:
            return self._player_info(mentioned_players[0])

        if (any(word in text for word in TOP_WORDS)
                or any(word in text for word in LIST_WORDS)
                or any(word in text for word in VALUE_WORDS)):
            return self._top(text, mentioned_teams)

        if any(word in text for word in GREETING_WORDS):
            return self._greeting()

        if mentioned_players:
            return self._player_info(mentioned_players[0])

        if mentioned_teams:
            return self._recommend(text, mentioned_teams)

        return self._fallback()

    # ----------------------------------------------------------- responses ----
    def _greeting(self) -> AssistantAnswer:
        return AssistantAnswer(
            reply=(
                "Здравей! Аз съм скаутинг асистентът на платформата. "
                "Мога да ти помогна да намериш подходящ футболист, да сравня двама играча, "
                "да покажа статистики или да препоръчам попълнение за отбор.\n\n"
                "Опитай например: *„препоръчай нападател за ЦСКА“* или *„сравни Маркус Думини и Йоан Коста“*."
            ),
            intent="greeting",
            suggestions=[
                "Топ 5 нападатели в лигата",
                "Препоръчай защитник за Локомотив",
                "Сравни Оливер Кръстев и Томас Гешев",
                "Кой е Маркус Думини",
            ],
        )

    def _help(self) -> AssistantAnswer:
        return AssistantAnswer(
            reply=(
                "Ето какво мога да направя:\n\n"
                "• **Търсене** – „покажи ми защитници под 25“\n"
                "• **Препоръки** – „препоръчай нападател за Витоша“\n"
                "• **Сравнение** – „сравни Рафинхо Сантос и Адриан Милчев“\n"
                "• **Статистика** – „топ реализатори в лигата“\n"
                "• **Профил** – „разкажи ми за Кристиян Златев“\n\n"
                "Всички отговори се изчисляват от реалните данни в базата и от обучения ML модел."
            ),
            intent="help",
            suggestions=[
                "Топ 10 футболисти по оценка",
                "Препоръчай полузащитник за ЦСКА",
                "Покажи ми всички вратари",
            ],
        )

    def _player_info(self, player: dict[str, Any]) -> AssistantAnswer:
        stats = self._stats_of(player["id"])
        lines = [
            f"**{player['name']}** – {POSITION_LABELS.get(player['position'], player['position'])}, "
            f"{player['team_name'] or 'без отбор'}.",
            f"Възраст: {player['age']} г. | Националност: {player['nationality']} | "
            f"Оценка: {player['overall']} (потенциал {player['potential']}) | "
            f"Пазарна стойност: {_money(player['market_value'])}",
        ]
        if stats:
            lines.append(
                f"Сезон {stats['season']}: {stats['appearances']} мача, "
                f"{stats['goals']} гола, {stats['assists']} асиста, "
                f"{stats['pass_accuracy']}% точност на пасовете."
            )
        return AssistantAnswer(
            reply="\n\n".join(lines),
            intent="player_info",
            cards=[_player_summary(player)],
            suggestions=[
                "Сравни го с най-подобния играч",
                "Покажи ми най-добрите в тази позиция",
            ],
        )

    def _compare(self, players: list[dict[str, Any]]) -> AssistantAnswer:
        if len(players) < 2:
            return AssistantAnswer(
                reply=(
                    "За сравнение ми трябват поне двама играча. Например: "
                    "*„сравни Маркус Думини и Йоан Коста“*."
                ),
                intent="compare",
            )

        selected = players[:4]
        ids = [int(player["id"]) for player in selected]
        rows = get_players(self.connection, ids)
        stats_rows = get_player_stats(self.connection, ids[0])
        comparison = []
        for player in rows:
            stats = get_player_stats(self.connection, int(player["id"]))
            latest = stats[-1] if stats else None
            summary = _player_summary(player)
            summary["stats"] = latest
            summary["stats_history"] = stats
            comparison.append(summary)

        first = comparison[0]
        latest = first.get("stats") or {}
        reply = (
            f"Сравнение между **{len(comparison)}** играча:\n\n"
            f"• {first['name']} – {first['position_label']}, оценка {first['overall']}, "
            f"стойност {_money(first['market_value'])}"
            + (f", {latest.get('goals', 0)} гола и {latest.get('assists', 0)} асиста в последния сезон"
               if latest else "")
            + "\n"
        )
        for player in comparison[1:]:
            reply += (
                f"• {player['name']} – {player['position_label']}, оценка {player['overall']}, "
                f"стойност {_money(player['market_value'])}"
                + (f", {player['stats'].get('goals', 0)} гола и {player['stats'].get('assists', 0)} асиста"
                   if player.get("stats") else "")
                + "\n"
            )
        reply += "\nОтвори пълното сравнение за детайлна таблица и радарна диаграма."

        return AssistantAnswer(
            reply=reply,
            intent="compare",
            comparison=comparison,
            meta={"ids": ids},
            suggestions=["Покажи подробен профил на първия", "Препоръчай подобни играчи"],
        )

    def _recommend(self, text: str, teams: list[dict[str, Any]]) -> AssistantAnswer:
        from ml.recommender import rank_players  # local import avoids a cycle
        from ml.train import MODEL_PATH
        from ml.model import ModelBundle

        model = None
        if MODEL_PATH.exists():
            try:
                model = ModelBundle.load(MODEL_PATH)
            except Exception:  # pragma: no cover - defensive
                model = None

        positions = _extract_positions(text)
        role, role_positions = positions
        position = role or "ST"

        age_min, age_max = _extract_age_filters(text)
        min_overall = _extract_number(text, (r"(\d{2})\s*\+",))

        candidates = search_players(
            self.connection,
            position=role_positions,
            age_min=age_min,
            age_max=age_max,
            min_overall=min_overall,
            limit=120,
        )
        if not candidates:
            candidates = search_players(self.connection, limit=120)

        stats_by_player = {}
        for candidate in candidates[:25]:
            rows = get_player_stats(self.connection, int(candidate["id"]))
            if rows:
                stats_by_player[int(candidate["id"])] = rows[-1]

        team_context = ""
        need = None
        if teams:
            team = teams[0]
            team_context = f" за **{team['name']}**"
            from database.repository import list_team_needs

            needs = list_team_needs(self.connection, int(team["id"]))
            # Only reuse a club request when it matches the requested role.
            for item in needs:
                if item["position"] == position:
                    need = item
                    break

        ranked = rank_players(
            candidates,
            position,
            stats_by_player=stats_by_player,
            model=model,
            need=need,
            limit=5,
            natural_positions_only=False,
        )

        reply = (
            f"Ето **{len(ranked)} препоръки**{team_context} "
            f"за ролята „{POSITION_LABELS.get(position, position)}“:\n\n"
        )
        for index, recommendation in enumerate(ranked, start=1):
            player = recommendation.player
            natural = POSITION_LABELS.get(player["position"], player["position"])
            suffix = "" if player["position"] == position else f" (родна позиция: {natural})"
            reply += (
                f"{index}. **{player['name']}**{suffix} ({player.get('team_name') or 'без отбор'}) – "
                f"скор {recommendation.score:.0f}/100, оценка {player['overall']}, "
                f"{_money(player['market_value'])}\n"
                f"   {_reason_summary(recommendation)}\n"
            )
        reply += "\nВсяка препоръка е придружена от обяснение защо играчът е избран."

        return AssistantAnswer(
            reply=reply,
            intent="recommend",
            cards=[_player_summary(item.player) for item in ranked],
            meta={"position": position, "team": teams[0]["name"] if teams else None},
            suggestions=["Сравни първите двама", "Покажи ми статистиката им"],
        )

    def _top(self, text: str, teams: list[dict[str, Any]]) -> AssistantAnswer:
        role, role_positions = _extract_positions(text)
        position = role

        limit = _extract_number(text, (r"топ\s*(\d{1,2})", r"(\d{1,2})\s*най-добр",)) or 0
        if limit == 0:
            limit = 10 if ("всички" in text or "покажи" in text) else 5
        limit = max(3, min(12, limit))

        sort = "overall"
        order = "desc"
        if any(word in text for word in ("скъпи", "цена", "стойност", "трансфер", "плат")):
            sort = "value"
        elif "най-млад" in text:
            sort = "age"
            order = "asc"
        elif "потенциал" in text:
            sort = "potential"

        age_min, age_max = _extract_age_filters(text)

        candidates = search_players(
            self.connection,
            position=role_positions,
            team_id=int(teams[0]["id"]) if teams else None,
            age_min=age_min,
            age_max=age_max,
            limit=limit,
            sort=sort,
            order=order,
        )
        if not candidates:
            candidates = search_players(
                self.connection,
                position=role_positions,
                team_id=int(teams[0]["id"]) if teams else None,
                age_min=age_min,
                age_max=age_max,
                limit=limit,
                sort=sort,
                order=order,
            )

        title = "Най-скъпите играчи" if sort == "value" else (
            "Най-младите играчи" if sort == "age" else "Топ класация"
        )
        reply = f"**{title}**" + (f" в отбора *{teams[0]['short_name']}*" if teams else "")
        reply += f" на позиция *{POSITION_LABELS.get(position, 'всички позиции')}*:\n\n"

        for index, player in enumerate(candidates, start=1):
            stats = self._stats_of(int(player["id"]))
            extra = ""
            if stats:
                extra = f" – {stats['goals']} гола, {stats['assists']} асиста (сезон {stats['season']})"
            reply += (
                f"{index}. **{player['name']}** ({player.get('team_name') or 'без отбор'}) – "
                f"оценка {player['overall']}, потенциал {player['potential']}, "
                f"{_money(player['market_value'])}{extra}\n"
            )

        return AssistantAnswer(
            reply=reply,
            intent="top",
            cards=[_player_summary(player) for player in candidates],
            suggestions=["Сравни първите двама", "Препоръчай подобни играчи"],
        )

    def _fallback(self) -> AssistantAnswer:
        return AssistantAnswer(
            reply=(
                "Не съм сигурен как да разбирам въпроса. Мога например:\n\n"
                "• да препоръчам играчи за конкретна позиция или отбор;\n"
                "• да сравня двама футболиста;\n"
                "• да покажа топ класацията по позиция;\n"
                "• да разкажа за конкретен играч.\n\n"
                "Опитай: *„препоръчай крило за Витоша“*"
            ),
            intent="fallback",
            suggestions=[
                "Топ 5 нападатели",
                "Препоръчай вратар за Лудогорец",
                "Кой е Кристиян Златев",
            ],
        )


def _money(value: float) -> str:
    value = float(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн. €"
    if value >= 1_000:
        return f"{value / 1_000:.0f} хил. €"
    return f"{value:.0f} €"


def _reason_summary(recommendation: Any) -> str:
    if recommendation.reasons:
        return "→ " + recommendation.reasons[0]
    return ""


def ask(question: str, connection: sqlite3.Connection) -> dict[str, Any]:
    """Convenience helper used by the Flask route."""
    assistant = ScoutingAssistant(connection)
    answer = assistant.answer(question)
    payload = answer.as_dict()
    payload["llm_available"] = llm_available()
    return payload
