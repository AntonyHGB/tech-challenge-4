"""Etapa 2 (2/3) — Predições do baseline sobre o Dataset de Produção.

Carrega o modelo treinado em `artifacts/model.joblib` (falha com mensagem clara
se ele não existir) e o aplica a um dataset de produção, gravando um CSV com a
`probabilidade_inadimplencia` e a `classe_prevista`. Se o rótulo
(`SeriousDlqin2yrs`) estiver presente, também reporta métricas no console.

Uso:
    python scripts/predizer.py
    python scripts/predizer.py --input data/production/producao_t3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credito.esquema import FEATURES, TARGET

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = RAIZ / "artifacts" / "model.joblib"
DEFAULT_INPUT = RAIZ / "data" / "production" / "producao_t1.csv"
DEFAULT_OUTPUT = RAIZ / "data" / "production" / "predicoes_t1.csv"

PROBABILIDADE = "probabilidade_inadimplencia"
CLASSE = "classe_prevista"
LIMITE = 0.5


def carregar_modelo(model_path: Path):
    """Carrega o modelo joblib; erro claro se o artefato não existir."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            "Rode antes: python scripts/treinar_modelo.py"
        )
    return joblib.load(model_path)


def predizer(input_csv: Path, model_path: Path, output_csv: Path) -> dict | None:
    """Gera prob/classe para o CSV e devolve as métricas (ou None sem rótulo)."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Dataset de produção não encontrado: {input_csv}. "
            "Rode antes: python scripts/simular_producao.py"
        )

    model = carregar_modelo(model_path)
    frame = pd.read_csv(input_csv)

    faltantes = [coluna for coluna in FEATURES if coluna not in frame.columns]
    if faltantes:
        raise ValueError(f"Colunas de feature ausentes no CSV: {faltantes}")

    probabilidades = model.predict_proba(frame[FEATURES])[:, 1]
    classes = (probabilidades >= LIMITE).astype(int)

    resultado = frame.copy()
    resultado[PROBABILIDADE] = probabilidades.round(6)
    resultado[CLASSE] = classes

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_csv, index=False)

    metricas = None
    if TARGET in frame.columns:
        rotulo = frame[TARGET]
        metricas = {
            "n": int(len(frame)),
            "taxa_prevista_positivos": round(float(classes.mean()), 4),
            "roc_auc": round(float(roc_auc_score(rotulo, probabilidades)), 4),
            "accuracy": round(float(accuracy_score(rotulo, classes)), 4),
            "precision": round(
                float(precision_score(rotulo, classes, zero_division=0)), 4
            ),
            "recall": round(float(recall_score(rotulo, classes, zero_division=0)), 4),
            "f1": round(float(f1_score(rotulo, classes, zero_division=0)), 4),
        }

    print("Predição do dataset de produção concluída:")
    print(f"  - modelo : {model_path}")
    print(f"  - entrada: {input_csv} ({len(frame)} linhas)")
    print(f"  - saída  : {output_csv}")
    print(f"  - colunas: '{PROBABILIDADE}', '{CLASSE}'")
    if metricas is not None:
        taxa = metricas["taxa_prevista_positivos"]
        print(f"  - taxa prevista (classe=1): {taxa:.4f}")
        print(
            "  - métricas (rótulo presente): "
            f"roc_auc={metricas['roc_auc']:.4f}, "
            f"accuracy={metricas['accuracy']:.4f}, "
            f"precision={metricas['precision']:.4f}, "
            f"recall={metricas['recall']:.4f}, "
            f"f1={metricas['f1']:.4f}"
        )
    else:
        print("  - sem rótulo: métricas não calculadas")
    return metricas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera predições do baseline para um dataset de produção."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        predizer(args.input, args.model, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
