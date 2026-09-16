"""Gera o lote CORROMPIDO usado para demonstrar o contrato bloqueando a ingestão.

Parte do dataset de referência (limpo) e reintroduz, de forma determinística,
os problemas que o contrato precisa barrar:

  1. idade <= 18
  2. renda nula (MonthlyIncome)
  3. linhas duplicadas
  4. valores fora de faixa (negativos e sentinelas 96/98)
  5. target inválido (fora de {0, 1})

Uso:
    python scripts/make_corrupted_batch.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schema import (  # noqa: E402
    PAST_DUE_COLUMNS,
    PAST_DUE_SENTINELS,
    TARGET,
)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "reference" / "reference.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corrupted" / "corrupted_batch.csv"

SEED = 7
N_INVALID_AGE = 15
N_NULL_INCOME = 150
N_DUPLICATES = 80
N_NEGATIVE = 60
N_BAD_PAST_DUE = 25
N_BAD_TARGET = 20


def make_corrupted(reference: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, list[str]]:
    """Aplica as corrupções e devolve (lote, relatório)."""
    rng = np.random.default_rng(seed)
    frame = reference.copy()
    report: list[str] = []

    # 1) Idade <= 18.
    rows = rng.choice(len(frame), size=N_INVALID_AGE, replace=False)
    frame.loc[frame.index[rows], "age"] = rng.choice([0, 15, 16, 17, 18], size=N_INVALID_AGE)
    report.append(f"idade <= 18                          : {N_INVALID_AGE} linhas")

    # 2) Renda nula.
    rows = rng.choice(len(frame), size=N_NULL_INCOME, replace=False)
    frame.loc[frame.index[rows], "MonthlyIncome"] = np.nan
    report.append(f"MonthlyIncome nula                   : {N_NULL_INCOME} linhas")

    # 3) Valores negativos (fora de faixa).
    rows = rng.choice(len(frame), size=N_NEGATIVE, replace=False)
    chosen = frame.index[rows]
    frame.loc[chosen, "MonthlyIncome"] = -rng.uniform(100, 5000, size=N_NEGATIVE).round(2)
    frame.loc[chosen, "RevolvingUtilizationOfUnsecuredLines"] = -rng.uniform(
        0.1, 2.0, size=N_NEGATIVE
    ).round(4)
    frame.loc[chosen, "DebtRatio"] = -rng.uniform(0.1, 3.0, size=N_NEGATIVE).round(4)
    frame.loc[chosen, "NumberOfDependents"] = -rng.integers(1, 4, size=N_NEGATIVE).astype(float)
    report.append(f"valores negativos (renda/utiliz/debt) : {N_NEGATIVE} linhas")

    # 4) Sentinelas 96/98 nos contadores de atraso (acima da faixa plausível).
    rows = rng.choice(len(frame), size=N_BAD_PAST_DUE, replace=False)
    chosen = frame.index[rows]
    sentinel = rng.choice(PAST_DUE_SENTINELS, size=N_BAD_PAST_DUE)
    for column in PAST_DUE_COLUMNS:
        frame.loc[chosen, column] = sentinel
    report.append(f"sentinela 96/98 nos contadores       : {N_BAD_PAST_DUE} linhas")

    # 5) Target inválido.
    rows = rng.choice(len(frame), size=N_BAD_TARGET, replace=False)
    frame.loc[frame.index[rows], TARGET] = rng.choice([2, -1], size=N_BAD_TARGET)
    report.append(f"target fora de {{0,1}}                 : {N_BAD_TARGET} linhas")

    # 6) Duplicatas exatas de linhas existentes.
    duplicated = frame.sample(n=N_DUPLICATES, random_state=seed)
    frame = pd.concat([frame, duplicated], ignore_index=True)
    report.append(f"linhas duplicadas (anexadas)         : {N_DUPLICATES} linhas")

    return frame, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o lote corrompido de demonstração.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"[erro] {args.input} não encontrado. Rode antes a preparação da referência.",
            file=sys.stderr,
        )
        return 2

    reference = pd.read_csv(args.input)
    corrupted, report = make_corrupted(reference, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corrupted.to_csv(args.output, index=False)

    print("Lote corrompido gerado:")
    print(f"  - referência: {args.input} ({len(reference)} linhas)")
    for line in report:
        print(f"  - {line}")
    print(f"  - saída     : {args.output} ({len(corrupted)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
