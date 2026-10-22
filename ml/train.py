"""Training entry point for the scouting model.

    python -m ml.train

The platform trains **one logistic-regression model per position group**
(GK / DEF / MID / ATT). A single league-wide model would learn the position of
the player instead of his quality, because the label ("did he deliver a strong
season?") is defined by goals and assists - a striker's speciality.

The trained models are written to ``ml/model.json`` and mirrored into the
``model_meta`` table so the web application can display the model metrics.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database.db import DEFAULT_DB_PATH, get_connection
from database.repository import get_model_meta, save_model_meta
from ml.features import (
    FEATURE_NAMES,
    POSITION_GROUP_LABELS,
    build_training_rows,
    team_strength_scores,
)
from ml.model import LogisticRegression, ModelBundle, binary_metrics, train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.json"
MODEL_NAME = "scouting-logistic"

POSITION_GROUPS = ("GK", "DEF", "MID", "ATT")
MIN_GROUP_SAMPLES = 12


def load_dataset(db_path: Path | str = DEFAULT_DB_PATH) -> tuple[list[dict], dict, list[dict]]:
    connection = get_connection(db_path)
    try:
        players = [
            dict(row)
            for row in connection.execute("SELECT * FROM players ORDER BY id").fetchall()
        ]
        stats_rows = [
            dict(row)
            for row in connection.execute("SELECT * FROM player_stats ORDER BY player_id, season").fetchall()
        ]
        team_rows = [
            dict(row)
            for row in connection.execute(
                """SELECT t.id, t.name, ROUND(AVG(p.overall), 1) AS avg_overall
                   FROM teams t LEFT JOIN players p ON p.team_id = t.id
                   GROUP BY t.id"""
            ).fetchall()
        ]
    finally:
        connection.close()

    stats_by_player: dict[int, list[dict]] = {}
    for row in stats_rows:
        stats_by_player.setdefault(row["player_id"], []).append(row)

    return players, stats_by_player, team_rows


def train_model(
    db_path: Path | str = DEFAULT_DB_PATH,
    model_path: Path | str = MODEL_PATH,
    save_to_db: bool = True,
) -> dict[str, Any]:
    """Train the model bundle, persist it and return a summary of the run."""
    players, stats_by_player, team_rows = load_dataset(db_path)
    strength = team_strength_scores(team_rows)

    X, y, rows = build_training_rows(players, stats_by_player, strength)
    if not X:
        raise RuntimeError("Няма данни за обучение - изпълнете `python -m database.seed`")

    models: dict[str, LogisticRegression] = {}
    group_metrics: dict[str, Any] = {}
    splits: dict[str, tuple[list, list, list, list]] = {}

    for group in POSITION_GROUPS:
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        if len(indices) < MIN_GROUP_SAMPLES:
            continue

        X_group = [X[index] for index in indices]
        y_group = [y[index] for index in indices]
        X_train, y_train, X_test, y_test = train_test_split(
            X_group, y_group, test_ratio=0.3, seed=42
        )

        model = LogisticRegression(learning_rate=0.18, iterations=5000, l2=0.01)
        model.fit(X_train, y_train, feature_names=FEATURE_NAMES)

        models[group] = model
        splits[group] = (X_train, y_train, X_test, y_test)
        group_metrics[group] = {
            "label": POSITION_GROUP_LABELS.get(group, group),
            "samples": len(X_group),
            "positives": sum(y_group),
            "train": model.evaluate(X_train, y_train),
            "test": model.evaluate(X_test, y_test),
        }

    if not models:
        raise RuntimeError("Няма достатъчно наблюдения за никоя позиционна група")

    def _aggregate(part: str) -> dict[str, float]:
        true_values: list[int] = []
        predictions: list[int] = []
        probabilities: list[float] = []
        for group, split in splits.items():
            model = models[group]
            X_part, y_part = (split[0], split[1]) if part == "train" else (split[2], split[3])
            group_probabilities = model.predict_proba(X_part)
            true_values.extend(y_part)
            probabilities.extend(group_probabilities)
            predictions.extend(1 if p >= 0.5 else 0 for p in group_probabilities)
        return binary_metrics(true_values, predictions, probabilities)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics: dict[str, Any] = {
        "train": _aggregate("train"),
        "test": _aggregate("test"),
        "samples": len(X),
        "positives": sum(y),
        "groups": group_metrics,
        "trained_at": trained_at,
    }

    bundle = ModelBundle(
        models=models,
        feature_names=list(FEATURE_NAMES),
        metrics=metrics,
        trained_at=trained_at,
    )
    bundle.save(model_path)

    weights: dict[str, Any] = {
        "features": list(FEATURE_NAMES),
        "groups": {
            group: {
                "label": POSITION_GROUP_LABELS.get(group, group),
                "coefficients": model.weights,
                "intercept": model.bias,
                "means": model.means,
                "stds": model.stds,
            }
            for group, model in models.items()
        },
    }

    if save_to_db:
        connection = get_connection(db_path)
        try:
            save_model_meta(connection, MODEL_NAME, trained_at, metrics, weights)
        finally:
            connection.close()

    return {"metrics": metrics, "weights": weights, "bundle": bundle, "models": models}


def model_report(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    connection = get_connection(db_path)
    try:
        return get_model_meta(connection, MODEL_NAME)
    finally:
        connection.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = train_model()
    metrics = result["metrics"]
    test = metrics["test"]
    train = metrics["train"]

    print("Моделите са обучени успешно.")
    print(f"  наблюдения: {metrics['samples']} (позитивни: {metrics['positives']})")
    print(f"  тренировка - accuracy: {train['accuracy']:.3f}, AUC: {train['auc']:.3f}, F1: {train['f1']:.3f}")
    print(f"  тест       - accuracy: {test['accuracy']:.3f}, AUC: {test['auc']:.3f}, F1: {test['f1']:.3f}")

    print("\n  Модел за всяка позиционна група:")
    for group, group_metrics in metrics.get("groups", {}).items():
        group_test = group_metrics["test"]
        print(
            f"    {group:<4} {group_metrics['label']:<15}"
            f" извадка: {group_metrics['samples']:>3}"
            f" · accuracy: {group_test['accuracy']:.3f}"
            f" · AUC: {group_test['auc']:.3f}"
            f" · F1: {group_test['f1']:.3f}"
        )

    print("\n  Най-силни коефициенти:")
    for group, model in result["models"].items():
        pairs = sorted(
            zip(model.feature_names, model.weights),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top = ", ".join(f"{name} ({value:+.2f})" for name, value in pairs[:4])
        print(f"    {group}: {top}")

    print(f"\n  моделът е записан в: {MODEL_PATH}")


if __name__ == "__main__":
    main()
