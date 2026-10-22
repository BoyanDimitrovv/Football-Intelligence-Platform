"""A small logistic-regression implementation written from scratch.

No external ML dependency is used on purpose: the platform has to install and
run offline, and a transparent model is much easier to explain during a
demonstration ("these are the learned weights and this is why").

The model is trained on season-level rows and predicts whether a player will
deliver a *high performance* season, which is then reused as one of the
components of the recommendation score.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass
class LogisticRegression:
    """Binary classifier trained with batch gradient descent + L2."""

    learning_rate: float = 0.15
    iterations: int = 4000
    l2: float = 0.0015
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ helpers ----
    def _standardize(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        return [
            [
                (value - self.means[index]) / (self.stds[index] if self.stds[index] else 1.0)
                for index, value in enumerate(row)
            ]
            for row in X
        ]

    # -------------------------------------------------------------- train ----
    def fit(self, X: list[list[float]], y: list[int],
            feature_names: Sequence[str] | None = None) -> "LogisticRegression":
        if not X or len(X) != len(y):
            raise ValueError("Невалидни тренировъчни данни")

        n_samples = len(X)
        n_features = len(X[0])
        self.feature_names = list(feature_names or [f"f{i}" for i in range(n_features)])

        self.means = [sum(row[i] for row in X) / n_samples for i in range(n_features)]
        variances = [
            sum((row[i] - self.means[i]) ** 2 for row in X) / n_samples
            for i in range(n_features)
        ]
        self.stds = [math.sqrt(v) if v > 1e-12 else 0.0 for v in variances]

        Z = self._standardize(X)
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.iterations):
            gradient_w = [0.0] * n_features
            gradient_b = 0.0
            for index, row in enumerate(Z):
                z = self.bias + sum(self.weights[j] * row[j] for j in range(n_features))
                prediction = sigmoid(z)
                error = prediction - y[index]
                for j in range(n_features):
                    gradient_w[j] += error * row[j]
                gradient_b += error
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * (
                    gradient_w[j] / n_samples + self.l2 * self.weights[j]
                )
            self.bias -= self.learning_rate * (gradient_b / n_samples)

        return self

    # ------------------------------------------------------------ predict ----
    def decision_function(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Raw logits (before the sigmoid).

        The sigmoid saturates for confident predictions, which makes the
        probability useless for ranking. The logit keeps the full resolution
        of the model and is therefore used for ordering.
        """
        Z = self._standardize([list(row) for row in X])
        return [
            self.bias + sum(self.weights[j] * row[j] for j in range(len(self.weights)))
            for row in Z
        ]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [sigmoid(value) for value in self.decision_function(X)]

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    # ----------------------------------------------------------- evaluate ----
    def evaluate(self, X: list[list[float]], y: list[int]) -> dict[str, float]:
        probabilities = self.predict_proba(X)
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        return binary_metrics(y, predictions, probabilities)

    # ------------------------------------------------------ serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "stds": self.stds,
            "feature_names": self.feature_names,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegression":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            learning_rate=payload.get("learning_rate", 0.15),
            iterations=payload.get("iterations", 4000),
            l2=payload.get("l2", 0.0015),
            weights=payload.get("weights", []),
            bias=payload.get("bias", 0.0),
            means=payload.get("means", []),
            stds=payload.get("stds", []),
            feature_names=payload.get("feature_names", []),
        )


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[float] | None = None,
) -> dict[str, float]:
    """Standard binary classification metrics (accuracy, precision, F1, AUC)."""
    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "auc": 0.0, "log_loss": 0.0, "samples": 0,
        }

    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    log_loss = 0.0
    if probabilities:
        log_loss = -sum(
            t * math.log(max(p, 1e-12)) + (1 - t) * math.log(max(1 - p, 1e-12))
            for p, t in zip(probabilities, y_true)
        ) / total

    # ROC AUC чрез ранговата статистика.
    scores = list(probabilities) if probabilities else [float(p) for p in y_pred]
    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 0
    for _, label in pairs:
        rank += 1
        if label == 1:
            rank_sum += rank
    positives = tp + fn
    negatives = tn + fp
    auc = 0.0
    if positives and negatives:
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "log_loss": round(log_loss, 4),
        "samples": total,
    }


def train_test_split(
    X: list[list[float]],
    y: list[int],
    test_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[list[float]], list[int], list[list[float]], list[int]]:
    """Deterministic shuffle split."""
    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * (1 - test_ratio))
    train_idx, test_idx = indices[:cut], indices[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, y_train, X_test, y_test


@dataclass
class ModelBundle:
    """Пакет от модели - по един за всяка позиционна група.

    Един общ модел за целия лига е изкривен: защото етикетът е изчислен
    според позицията, нападателите „изнасят" и коефициентите се настройват
    към тях. Един модел на позиционна група учи какво прави един играч
    добър **сред своите връстници**, което е и правилната интерпретация на
    „ще има силен сезон".
    """

    models: dict[str, LogisticRegression] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""

    # ------------------------------------------------------------ helpers ----
    def for_group(self, group: str) -> LogisticRegression | None:
        """Моделът за дадена позиционна група (или None)."""
        model = self.models.get(group)
        if model is not None and model.weights:
            return model
        return None

    def logit(
        self,
        player: dict[str, Any],
        stats: dict[str, Any] | None = None,
        team_strength: float = 1.0,
        group: str | None = None,
    ) -> float:
        """Суровият логит на модела за групата на играча."""
        from ml.features import player_features, position_group_of

        resolved = group or position_group_of(player)
        model = self.for_group(resolved)
        if model is None:
            return 0.0
        features = player_features(player, stats, team_strength)
        return model.decision_function([features])[0]

    @property
    def groups(self) -> list[str]:
        return sorted(self.models)

    # ------------------------------------------------------ serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "model-bundle",
            "feature_names": self.feature_names,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "models": {
                group: model.to_dict() for group, model in self.models.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        models: dict[str, LogisticRegression] = {}
        for group, data in (payload.get("models") or {}).items():
            models[group] = LogisticRegression(
                learning_rate=data.get("learning_rate", 0.15),
                iterations=data.get("iterations", 4000),
                l2=data.get("l2", 0.0015),
                weights=data.get("weights", []),
                bias=data.get("bias", 0.0),
                means=data.get("means", []),
                stds=data.get("stds", []),
                feature_names=data.get("feature_names", []),
            )
        return cls(
            models=models,
            feature_names=payload.get("feature_names", []),
            metrics=payload.get("metrics", {}),
            trained_at=payload.get("trained_at", ""),
        )

