"""Seed data for the Football Intelligence Platform.

The dataset below is a **synthetic demo dataset**: clubs, players and all
statistics are generated for demonstration purposes only. It is fully
deterministic (fixed random seed) so that every run of the application shows
the same numbers and the trained model can be reproduced.

Run directly to (re)build the database:

    python -m database.seed
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from database.db import DEFAULT_DB_PATH, get_connection, init_db

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_EXPORT_PATH = PROJECT_DIR / "data" / "raw" / "players.json"

SEASONS = ("2021/22", "2022/23", "2023/24", "2024/25")

# Attribute order used by the compact player tuples below.
ATTR_FIELDS = (
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physical",
    "vision",
    "positioning",
    "composure",
    "teamwork",
    "stamina",
    "discipline",
    "form",
    "fitness",
    "injury_risk",
)

TEAMS: list[dict[str, Any]] = [
    {"name": "Лудогорец", "short_name": "Лудогорец", "league": "Българска първа лига",
     "country": "България", "stadium": "Хюсеид Али", "color": "#eab308"},
    {"name": "ЦСКА", "short_name": "ЦСКА", "league": "Българска първа лига",
     "country": "България", "stadium": "Балкан", "color": "#f43f5e"},
    {"name": "Витоша", "short_name": "Витоша", "league": "Българска първа лига",
     "country": "България", "stadium": "Стадион Витоша", "color": "#22c55e"},
    {"name": "Локомотив", "short_name": "Локомотив", "league": "Българска първа лига",
     "country": "България", "stadium": "Стадион Локомотив", "color": "#f97316"},
    {"name": "Ботев", "short_name": "Ботев", "league": "Българска първа лига",
     "country": "България", "stadium": "Христо Ботев", "color": "#ef4444"},
    {"name": "Черно море", "short_name": "Черно море", "league": "Българска първа лига",
     "country": "България", "stadium": "Славянски", "color": "#a855f7"},
    {"name": "Славяне", "short_name": "Славяне", "league": "Българска първа лига",
     "country": "България", "stadium": "Славяне", "color": "#38bdf8"},
    {"name": "Спартак", "short_name": "Спартак", "league": "Българска първа лига",
     "country": "България", "stadium": "Спартак", "color": "#14b8a6"},
]

# name, position, nationality, age, team_id, height, foot, value, overall, potential, attributes
PLAYERS: list[tuple[Any, ...]] = [
    ("Иван Стоянов", "GK", "България", 31, 1, 188, "ляво", 2_500_000, 78, 79,
     (52, 32, 66, 38, 40, 74, 58, 80, 78, 72, 60, 76, 68, 70, 14)),
    ("Мартин Петков", "GK", "България", 24, 4, 190, "дясно", 1_200_000, 71, 77,
     (55, 30, 62, 35, 38, 70, 54, 72, 70, 68, 62, 72, 74, 76, 12)),
    ("Георги Димитров", "GK", "България", 27, 6, 186, "дясно", 900_000, 73, 75,
     (50, 30, 70, 36, 40, 72, 62, 76, 72, 70, 58, 74, 66, 71, 18)),
    ("Даниел Николов", "GK", "България", 19, 7, 193, "дясно", 350_000, 64, 84,
     (58, 28, 64, 42, 34, 68, 58, 70, 66, 66, 60, 68, 72, 80, 10)),
    ("Сен Атанасе", "GK", "Гана", 25, 5, 191, "дясно", 1_100_000, 70, 76,
     (52, 30, 66, 38, 36, 74, 56, 72, 70, 70, 62, 70, 72, 76, 18)),

    ("Тодор Василев", "CB", "България", 28, 2, 187, "дясно", 3_200_000, 79, 80,
     (62, 38, 66, 46, 86, 82, 60, 84, 78, 76, 74, 82, 70, 74, 22)),
    ("Николай Петков", "CB", "България", 26, 3, 186, "ляво", 2_800_000, 77, 79,
     (64, 36, 68, 44, 84, 80, 58, 82, 76, 78, 72, 84, 74, 76, 18)),
    ("Радослав Трифонов", "CB", "България", 33, 5, 188, "дясно", 1_400_000, 73, 74,
     (58, 40, 62, 42, 78, 76, 55, 76, 74, 72, 68, 86, 62, 62, 32)),
    ("Ервин Муса", "CB", "България", 25, 6, 190, "ляво", 2_100_000, 75, 79,
     (68, 34, 64, 46, 80, 78, 56, 78, 74, 75, 74, 76, 70, 78, 20)),
    ("Ясен ван Дер Berg", "CB", "Нидерландия", 27, 6, 189, "дясно", 3_000_000, 75, 77,
     (64, 36, 70, 48, 82, 78, 58, 80, 76, 78, 74, 74, 68, 76, 20)),

    ("Кристофер Манчев", "LB", "България", 23, 2, 179, "ляво", 1_800_000, 72, 80,
     (82, 44, 68, 72, 72, 70, 62, 76, 68, 74, 80, 74, 76, 82, 14)),
    ("Петър Колев", "LB", "България", 24, 7, 178, "ляво", 1_100_000, 68, 76,
     (78, 42, 66, 68, 68, 68, 58, 70, 66, 70, 78, 70, 72, 78, 16)),
    ("Виктор Недялков", "RB", "България", 29, 5, 180, "дясно", 1_600_000, 71, 72,
     (74, 46, 68, 64, 72, 72, 58, 72, 70, 72, 76, 78, 68, 70, 22)),
    ("Стоян Ангелов", "RB", "България", 21, 1, 177, "дясно", 700_000, 66, 80,
     (84, 42, 64, 70, 64, 66, 56, 68, 66, 68, 78, 72, 76, 84, 12)),

    ("Оливер Кръстев", "CDM", "България", 30, 4, 181, "дясно", 3_800_000, 80, 81,
     (64, 52, 78, 58, 82, 80, 70, 78, 78, 84, 82, 80, 68, 68, 28)),
    ("Явор Иванов", "CDM", "България", 22, 3, 183, "дясно", 1_900_000, 73, 84,
     (74, 50, 76, 64, 74, 72, 64, 74, 72, 76, 80, 76, 80, 84, 12)),
    ("Лука Йованович", "CDM", "Сърбия", 24, 1, 186, "дясно", 3_400_000, 76, 81,
     (70, 52, 76, 66, 78, 82, 66, 78, 76, 76, 78, 80, 74, 82, 16)),

    ("Томас Гешев", "CM", "България", 27, 1, 180, "дясно", 4_500_000, 81, 83,
     (68, 62, 82, 72, 66, 72, 78, 78, 80, 82, 84, 72, 78, 78, 16)),
    ("Адриан Милчев", "CM", "България", 26, 2, 176, "ляво", 3_600_000, 79, 81,
     (72, 64, 80, 74, 62, 68, 76, 76, 78, 80, 80, 74, 82, 82, 14)),
    ("Рафинхо Сантос", "CM", "Бразилия", 26, 2, 178, "дясно", 4_800_000, 80, 82,
     (74, 70, 82, 80, 58, 72, 78, 76, 80, 78, 82, 70, 80, 82, 14)),
    ("Виктор Радков", "CM", "България", 20, 5, 179, "дясно", 900_000, 67, 83,
     (76, 58, 76, 76, 58, 66, 70, 68, 72, 72, 78, 70, 82, 86, 10)),
    ("Стивен Манов", "CAM", "България", 34, 4, 175, "дясно", 900_000, 70, 70,
     (56, 68, 78, 66, 44, 62, 76, 72, 74, 76, 64, 70, 58, 54, 38)),
    ("Никита Иванов", "CAM", "България", 25, 6, 178, "дясно", 4_100_000, 78, 82,
     (70, 74, 78, 78, 48, 64, 82, 76, 78, 74, 76, 66, 80, 80, 14)),

    ("Асен Димитров", "LW", "България", 24, 7, 178, "дясно", 2_600_000, 76, 82,
     (88, 74, 70, 84, 38, 62, 76, 78, 74, 70, 82, 64, 84, 84, 12)),
    ("Мартин Стоянов", "LW", "България", 28, 3, 181, "ляво", 2_200_000, 73, 75,
     (80, 72, 72, 78, 42, 66, 70, 74, 74, 72, 78, 72, 66, 70, 20)),

    ("Давид Христов", "RW", "България", 22, 1, 176, "дясно", 2_900_000, 77, 85,
     (90, 78, 68, 86, 36, 60, 74, 76, 76, 68, 84, 60, 86, 86, 10)),
    ("Емил Салиев", "RW", "България", 23, 2, 175, "ляво", 2_400_000, 75, 83,
     (86, 76, 70, 84, 38, 62, 72, 74, 74, 70, 82, 66, 84, 86, 12)),
    ("Диего Мартин", "RW", "Испания", 23, 5, 176, "ляво", 3_100_000, 76, 84,
     (86, 76, 72, 88, 36, 62, 74, 76, 74, 68, 80, 64, 84, 86, 12)),
    ("Силвестър Петков", "RW", "България", 29, 6, 177, "дясно", 1_500_000, 71, 72,
     (76, 70, 68, 74, 44, 68, 68, 70, 72, 70, 74, 76, 62, 66, 24)),

    ("Маркус Думини", "ST", "България", 27, 1, 183, "дясно", 7_500_000, 84, 86,
     (82, 88, 68, 78, 32, 82, 72, 86, 84, 76, 80, 70, 86, 82, 20)),
    ("Йоан Коста", "ST", "България", 25, 3, 185, "дясно", 6_200_000, 81, 85,
     (78, 85, 66, 76, 30, 84, 74, 84, 82, 74, 78, 72, 82, 84, 16)),
    ("Кристиян Златев", "ST", "България", 21, 5, 184, "дясно", 3_200_000, 74, 86,
     (80, 82, 62, 74, 26, 76, 68, 80, 76, 70, 78, 62, 84, 88, 10)),
    ("Иван Романов", "ST", "България", 32, 7, 186, "дясно", 1_800_000, 72, 72,
     (66, 76, 64, 62, 36, 80, 64, 78, 74, 72, 70, 74, 60, 60, 34)),
    ("Фредрик Ели", "ST", "България", 26, 6, 182, "дясно", 4_800_000, 79, 82,
     (84, 82, 68, 74, 34, 80, 70, 80, 78, 72, 80, 68, 78, 80, 18)),
    ("Елинд Окот", "ST", "Кения", 22, 3, 180, "дясно", 2_000_000, 74, 85,
     (90, 80, 62, 76, 28, 78, 62, 78, 70, 72, 80, 60, 86, 88, 12)),
    ("Ангел Димитров", "ST", "България", 19, 2, 181, "дясно", 1_600_000, 69, 87,
     (86, 80, 60, 78, 24, 72, 66, 76, 72, 66, 74, 58, 84, 90, 8)),
]

# team_id -> list of (position, priority, min_overall, max_age, max_value, notes)
TEAM_NEEDS: dict[int, list[tuple[Any, ...]]] = {
    1: [("ST", 1, 80, 30, 12_000_000, "Липсва първи нападател с обем в штрафната зона"),
        ("LB", 2, 72, 27, 3_500_000, "Търси се бърз ляв защитник за висока линия"),
        ("CAM", 3, 76, 26, 6_000_000, "Нужен е креативен организатор под нападателя")],
    2: [("RW", 1, 76, 25, 5_000_000, "Дясно крило с дрифт и 1-на-1"),
        ("CB", 2, 76, 29, 6_000_000, "Водещ защитник за централна двойка"),
        ("GK", 3, 72, 28, 4_000_000, "По-сигурна алтернатива на вратаря")],
    3: [("ST", 1, 79, 28, 9_000_000, "Нужен е реализатор с обем пред вратаря"),
        ("CB", 2, 74, 30, 5_000_000, "Опитен защитник за ниска отбрана"),
        ("LW", 3, 72, 26, 3_500_000, "Скоростен нападач отляво")],
    4: [("CDM", 1, 78, 31, 6_000_000, "Опорен полузащитник с опит в-лигата"),
        ("CM", 2, 76, 30, 6_000_000, "Пазител на ритъма в средата на терена")],
    5: [("ST", 1, 75, 26, 4_500_000, "Млад нападател за бъдещето"),
        ("RW", 2, 74, 25, 4_000_000, "Дясно крило за ротация в атаката")],
    6: [("GK", 1, 73, 27, 3_000_000, "По-млад вратар с добро позициониране"),
        ("CM", 2, 76, 28, 6_000_000, "Контролен играч от втори етаж"),
        ("LB", 3, 70, 26, 3_000_000, "Енергичен ляв защитник")],
    7: [("ST", 1, 76, 26, 5_000_000, "Инвазивен форвард от млада сметка"),
        ("RW", 2, 74, 25, 4_000_000, "Скорост за контраатаки")],
    8: [("CB", 1, 73, 30, 3_500_000, "Стабилен защитник за опитна среда"),
        ("CM", 2, 72, 29, 3_500_000, "Полузащитник с универсалност")],
}

# Per-90 production baselines per position.
POSITION_OUTPUT = {
    "GK":  {"goals": 0.00, "assists": 0.00, "shots": 0.10, "key_passes": 0.05, "tackles": 0.10,
            "interceptions": 0.05, "pass_accuracy": 82.0, "dribbles": 0.2, "clean_sheets": 0.36},
    "CB":  {"goals": 0.03, "assists": 0.02, "shots": 0.45, "key_passes": 0.45, "tackles": 1.85,
            "interceptions": 1.35, "pass_accuracy": 73.0, "dribbles": 0.8, "clean_sheets": 0.22},
    "LB":  {"goals": 0.05, "assists": 0.09, "shots": 0.70, "key_passes": 1.10, "tackles": 2.05,
            "interceptions": 1.45, "pass_accuracy": 75.0, "dribbles": 1.4, "clean_sheets": 0.18},
    "RB":  {"goals": 0.05, "assists": 0.09, "shots": 0.70, "key_passes": 1.10, "tackles": 2.05,
            "interceptions": 1.45, "pass_accuracy": 75.0, "dribbles": 1.4, "clean_sheets": 0.18},
    "CDM": {"goals": 0.04, "assists": 0.06, "shots": 0.85, "key_passes": 1.00, "tackles": 2.45,
            "interceptions": 2.05, "pass_accuracy": 79.0, "dribbles": 1.0, "clean_sheets": 0.20},
    "CM":  {"goals": 0.08, "assists": 0.13, "shots": 1.35, "key_passes": 1.70, "tackles": 1.85,
            "interceptions": 1.60, "pass_accuracy": 81.0, "dribbles": 1.6, "clean_sheets": 0.20},
    "CAM": {"goals": 0.16, "assists": 0.21, "shots": 1.90, "key_passes": 2.40, "tackles": 0.75,
            "interceptions": 0.95, "pass_accuracy": 79.0, "dribbles": 2.4, "clean_sheets": 0.18},
    "LW":  {"goals": 0.21, "assists": 0.22, "shots": 2.20, "key_passes": 1.75, "tackles": 0.70,
            "interceptions": 0.60, "pass_accuracy": 74.0, "dribbles": 2.9, "clean_sheets": 0.16},
    "RW":  {"goals": 0.21, "assists": 0.22, "shots": 2.20, "key_passes": 1.75, "tackles": 0.70,
            "interceptions": 0.60, "pass_accuracy": 74.0, "dribbles": 2.9, "clean_sheets": 0.16},
    "ST":  {"goals": 0.46, "assists": 0.11, "shots": 3.10, "key_passes": 0.95, "tackles": 0.45,
            "interceptions": 0.35, "pass_accuracy": 71.0, "dribbles": 1.7, "clean_sheets": 0.14},
}


def _player_row(record: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a compact player tuple into a flat dictionary."""
    (name, position, nationality, age, team_id, height, foot, value,
     overall, potential, attributes) = record
    row: dict[str, Any] = {
        "name": name,
        "position": position,
        "nationality": nationality,
        "age": age,
        "team_id": team_id,
        "height_cm": height,
        "preferred_foot": foot,
        "market_value": value,
        "overall": overall,
        "potential": potential,
        "contract_years": max(1, min(5, 22 + age // 6 - age // 12)),
    }
    row.update(dict(zip(ATTR_FIELDS, attributes)))
    return row


def _season_stats(player: dict[str, Any], season_index: int, rng: random.Random) -> dict[str, Any] | None:
    """Generate one consistent season line for a player."""
    age_in_season = player["age"] - (len(SEASONS) - 1 - season_index)
    if age_in_season < 17:
        return None

    output = POSITION_OUTPUT[player["position"]]
    overall = player["overall"]

    # Young players get fewer minutes, veterans slow down a little.
    if age_in_season <= 19:
        base_apps = 13
    elif age_in_season <= 21:
        base_apps = 20
    elif age_in_season >= 34:
        base_apps = 24
    else:
        base_apps = 28
    base_apps += season_index  # more established with every season
    fitness_mod = (player["fitness"] - 70) / 12
    appearances = int(round(base_apps + fitness_mod + rng.uniform(-3, 3)))
    appearances = max(6, min(34, appearances))

    minutes = int(round(appearances * rng.uniform(76, 91)))

    # Output scales with the player's quality and their career progression.
    quality = 0.72 + overall / 100.0 * 0.55
    progression = 0.88 + 0.06 * season_index
    if age_in_season <= 20:
        progression += 0.05
    factor = quality * progression

    ninetys = minutes / 90.0
    goals = max(0, round(ninetys * output["goals"] * factor + rng.uniform(-0.4, 0.4)))
    assists = max(0, round(ninetys * output["assists"] * factor + rng.uniform(-0.3, 0.3)))

    shot_quality = 0.55 + player["shooting"] / 250.0
    shots = max(0, round(ninetys * output["shots"] * (0.75 + player["shooting"] / 220.0) + rng.uniform(-1, 1)))
    shots_on_target = int(round(shots * shot_quality))

    key_passes = max(0, round(ninetys * output["key_passes"] * (0.7 + player["passing"] / 260.0) + rng.uniform(-1, 1)))
    tackles = max(0, round(ninetys * output["tackles"] * (0.65 + player["defending"] / 200.0) + rng.uniform(-2, 2)))
    interceptions = max(0, round(ninetys * output["interceptions"] * (0.65 + player["positioning"] / 240.0) + rng.uniform(-2, 2)))
    dribbles = max(0, round(ninetys * output["dribbles"] * (0.5 + player["dribbling"] / 130.0) + rng.uniform(-1, 1)))

    pass_accuracy = round(
        output["pass_accuracy"] + (player["passing"] - 70) * 0.28 + rng.uniform(-2.5, 2.5), 1
    )
    pass_accuracy = max(58.0, min(94.0, pass_accuracy))

    duels_won = round(44 + (player["physical"] - 65) * 0.45 + rng.uniform(-4, 4), 1)
    duels_won = max(30.0, min(70.0, duels_won))

    clean_sheets = 0
    if player["position"] == "GK":
        clean_sheets = int(round(appearances * output["clean_sheets"] * (0.7 + player["positioning"] / 220.0)))
    elif player["position"] in ("CB", "LB", "RB", "CDM"):
        clean_sheets = int(round(appearances * output["clean_sheets"] * rng.uniform(0.2, 1.0)))

    yellow = int(round(appearances * (0.42 - (player["discipline"] - 70) * 0.02) * rng.uniform(0.6, 1.4)))
    yellow = max(0, yellow)
    red = 1 if rng.random() < max(0.0, 0.10 - (player["discipline"] - 70) * 0.008) else 0

    return {
        "season": SEASONS[season_index],
        "competition": "Българска първа лига",
        "appearances": appearances,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "shots": shots,
        "shots_on_target": shots_on_target,
        "key_passes": key_passes,
        "tackles": tackles,
        "interceptions": interceptions,
        "duels_won_pct": duels_won,
        "pass_accuracy": pass_accuracy,
        "dribbles": dribbles,
        "clean_sheets": clean_sheets,
        "yellow_cards": yellow,
        "red_cards": red,
    }


def seed_database(db_path: Path | str = DEFAULT_DB_PATH, reset: bool = True) -> None:
    """Populate the database with the demo dataset."""
    db_path = Path(db_path)
    init_db(db_path)
    connection = get_connection(db_path)
    try:
        if reset:
            connection.executescript(
                "DELETE FROM player_stats; DELETE FROM team_needs; DELETE FROM players; DELETE FROM teams;"
            )
            connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('players','teams','player_stats','team_needs')")

        for team in TEAMS:
            connection.execute(
                """INSERT INTO teams (name, short_name, league, country, stadium, color)
                   VALUES (:name, :short_name, :league, :country, :stadium, :color)""",
                team,
            )
        connection.commit()

        for record in PLAYERS:
            row = _player_row(record)
            columns = ", ".join(row.keys())
            placeholders = ", ".join(f":{key}" for key in row)
            cursor = connection.execute(
                f"INSERT INTO players ({columns}) VALUES ({placeholders})", row
            )
            player_id = cursor.lastrowid

            rng = random.Random(player_id * 7919 + 17)
            for season_index in range(len(SEASONS)):
                stats = _season_stats(row, season_index, rng)
                if stats is None:
                    continue
                stats["player_id"] = player_id
                stat_columns = ", ".join(stats.keys())
                stat_placeholders = ", ".join(f":{key}" for key in stats)
                connection.execute(
                    f"INSERT INTO player_stats ({stat_columns}) VALUES ({stat_placeholders})", stats
                )
        connection.commit()

        for team_id, needs in TEAM_NEEDS.items():
            for position, priority, min_overall, max_age, max_value, notes in needs:
                connection.execute(
                    """INSERT INTO team_needs
                       (team_id, position, priority, min_overall, max_age, max_value, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (team_id, position, priority, min_overall, max_age, max_value, notes),
                )
        connection.commit()
    finally:
        connection.close()


def export_dataset(db_path: Path | str = DEFAULT_DB_PATH,
                   export_path: Path | str = RAW_EXPORT_PATH) -> Path:
    """Export the seeded dataset as JSON into data/raw/ for the data pipeline."""
    db_path = Path(db_path)
    connection = get_connection(db_path)
    try:
        payload: dict[str, Any] = {"teams": [], "players": []}
        payload["teams"] = [
            dict(row) for row in connection.execute("SELECT * FROM teams ORDER BY id").fetchall()
        ]
        for row in connection.execute(
            """SELECT p.*, t.name AS team_name
               FROM players p LEFT JOIN teams t ON t.id = p.team_id
               ORDER BY p.id"""
        ).fetchall():
            player = dict(row)
            player["stats"] = [
                dict(s) for s in connection.execute(
                    "SELECT * FROM player_stats WHERE player_id = ? ORDER BY season",
                    (player["id"],),
                ).fetchall()
            ]
            payload["players"].append(player)
    finally:
        connection.close()

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return export_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db_path = DEFAULT_DB_PATH
    seed_database(db_path)
    export = export_dataset(db_path)
    connection = get_connection(db_path)
    try:
        players = connection.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        stats = connection.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0]
    finally:
        connection.close()
    print(f"Базата данни е готова: {db_path}")
    print(f"  футболисти: {players} | сезонни статистики: {stats}")
    print(f"  JSON експорт: {export}")


if __name__ == "__main__":
    main()
