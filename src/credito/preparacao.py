"""Etapa 1 (2/5) — Construção do dataset de REFERÊNCIA (limpo).

O dataset de referência é o único usado no treino do baseline. Ele é limpo
para satisfazer o contrato de dados (`credito/contrato_dados.py`):

  1. remove a coluna de índice do CSV original;
  2. remove linhas com valores ausentes (MonthlyIncome / NumberOfDependents);
  3. remove linhas duplicadas;
  4. remove idades <= 18 (registro inválido no original);
  5. remove os códigos de sentinela 96/98 dos contadores de atraso.

Uso:
    python scripts/preparar_referencia.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credito.esquema import (
    COLUMNS,
    MAX_PAST_DUE,
    MIN_AGE_EXCLUSIVE,
    PAST_DUE_COLUMNS,
    PAST_DUE_SENTINELS,
    RAW_INDEX_COLUMN,
)

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ / "data" / "raw" / "give_me_some_credit_training.csv"
DEFAULT_OUTPUT = RAIZ / "data" / "reference" / "reference.csv"


def preparar_referencia(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Limpa o CSV bruto e grava o dataset de referência."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"{input_csv} não encontrado. Rode antes: python scripts/baixar_dataset.py"
        )

    frame = pd.read_csv(input_csv)
    report: list[str] = [f"entrada : {len(frame)} linhas x {frame.shape[1]} colunas"]

    # 1) Índice do CSV original.
    if RAW_INDEX_COLUMN in frame.columns:
        frame = frame.drop(columns=[RAW_INDEX_COLUMN])
        report.append(f"coluna '{RAW_INDEX_COLUMN}' removida")

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV de entrada: {missing}")
    frame = frame[COLUMNS]

    # 2) Valores ausentes.
    nulls_before = int(frame.isna().sum().sum())
    frame = frame.dropna(subset=COLUMNS)
    report.append(f"nulos removidos: {nulls_before}")

    # 3) Duplicatas.
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    report.append(f"duplicatas removidas: {duplicates}")

    # 4) Idade > 18.
    invalid_age = int((frame["age"] <= MIN_AGE_EXCLUSIVE).sum())
    frame = frame[frame["age"] > MIN_AGE_EXCLUSIVE]
    report.append(f"idades <= {MIN_AGE_EXCLUSIVE} removidas: {invalid_age}")

    # 5) Sentinela 96/98 nos contadores de atraso.
    sentinel_mask = pd.Series(False, index=frame.index)
    for column in PAST_DUE_COLUMNS:
        sentinel_mask |= frame[column].isin(PAST_DUE_SENTINELS)
    sentinels = int(sentinel_mask.sum())
    frame = frame[~sentinel_mask]
    report.append(f"linhas com sentinela {PAST_DUE_SENTINELS} removidas: {sentinels}")

    # Os contadores de atraso são inteiros, mas vêm como float quando há nulos.
    frame = frame.copy()
    for column in PAST_DUE_COLUMNS:
        frame[column] = frame[column].astype(int)

    frame = frame.reset_index(drop=True)
    report.append(f"saída   : {len(frame)} linhas x {frame.shape[1]} colunas")

    if frame.empty:
        raise ValueError("Dataset de referência ficou vazio após a limpeza.")

    max_past_due = int(frame[PAST_DUE_COLUMNS].to_numpy().max())
    if max_past_due > MAX_PAST_DUE:
        raise ValueError(
            f"Limpeza inconsistente: contador de atraso máximo = {max_past_due} "
            f"(contrato exige <= {MAX_PAST_DUE})."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    print("Preparação da referência concluída:")
    for line in report:
        print(f"  - {line}")
    print(
        f"  - taxa de inadimplência (target=1): {frame['SeriousDlqin2yrs'].mean():.4f}"
    )
    print(f"  - arquivo: {output_csv}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o dataset de referência limpo.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    preparar_referencia(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
