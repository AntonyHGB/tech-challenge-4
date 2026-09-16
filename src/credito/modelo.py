"""Etapa 1 (5/5) — Treino do classificador binário baseline de risco de crédito.

Treina uma Regressão Logística sobre o dataset de REFERÊNCIA (limpo), com
split estratificado treino/teste, e reporta AUC, precision, recall, accuracy e
F1. O modelo e as métricas são gravados em `artifacts/`.

Uso:
    python scripts/treinar_modelo.py
    python scripts/treinar_modelo.py --input data/reference/reference.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credito.esquema import FEATURES, TARGET

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_MODEL = RAIZ / "artifacts" / "model.joblib"
DEFAULT_METRICS = RAIZ / "artifacts" / "baseline_metrics.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def construir_modelo() -> Pipeline:
    """Pipeline baseline: imputação -> padronização -> regressão logística."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def treinar(input_csv: Path, model_path: Path, metrics_path: Path) -> dict:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"{input_csv} não encontrado. Rode antes: python scripts/baixar_dataset.py "
            "e python scripts/preparar_referencia.py"
        )

    frame = pd.read_csv(input_csv)
    features = frame[FEATURES]
    target = frame[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = construir_modelo()
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    try:
        dataset_name = str(input_csv.relative_to(RAIZ))
    except ValueError:
        # --input relativo: relative_to exige subcaminho textual de RAIZ.
        dataset_name = str(input_csv)

    metrics = {
        "dataset": dataset_name,
        "n_total": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": round(float(target.mean()), 4),
        "model": "LogisticRegression(class_weight='balanced', max_iter=1000)",
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    print("Treino do baseline concluído:")
    print(
        f"  - amostras      : {metrics['n_total']} "
        f"(treino={metrics['n_train']}, teste={metrics['n_test']})"
    )
    print(f"  - taxa positivos: {metrics['positive_rate']:.4f}")
    print(f"  - ROC AUC       : {metrics['roc_auc']:.4f}")
    print(f"  - accuracy      : {metrics['accuracy']:.4f}")
    print(f"  - precision     : {metrics['precision']:.4f}")
    print(f"  - recall        : {metrics['recall']:.4f}")
    print(f"  - F1            : {metrics['f1']:.4f}")
    print(f"  - matriz confusão ([[TN, FP], [FN, TP]]): {metrics['confusion_matrix']}")
    print("  - relatório de classificação:")
    print(classification_report(y_test, predictions, digits=4))
    print(f"  - modelo  : {model_path}")
    print(f"  - métricas: {metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treina o classificador baseline de risco."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    treinar(args.input, args.model, args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
