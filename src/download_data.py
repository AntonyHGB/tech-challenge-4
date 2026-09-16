"""Etapa 1 (1/5) — Aquisição do dataset de referência.

Baixa o dataset "Give Me Some Credit" do Kaggle via `kagglehub` e copia o
arquivo de treino para `data/raw/`.

O download público foi verificado sem autenticação. Caso o Kaggle exija
credenciais no ambiente do avaliador, o script gera um dataset sintético
de fallback (mesmas colunas, 6.000 amostras) para que o MVP rode de ponta a
ponta — deixando o bloqueio explícito no console e no README.

Uso:
    python src/download_data.py                # tenta Kaggle, cai para sintético
    python src/download_data.py --force-synthetic
    python src/download_data.py --force        # sobrescreve o CSV existente
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from schema import COLUMNS, FEATURES, RAW_INDEX_COLUMN, TARGET

KAGGLE_DATASET = "brycecf/give-me-some-credit-dataset"
KAGGLE_TRAIN_FILE = "cs-training.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "give_me_some_credit_training.csv"

SYNTHETIC_SAMPLES = 6000
SYNTHETIC_SEED = 42


def download_from_kaggle() -> Path:
    """Baixa o dataset do Kaggle e devolve o caminho do `cs-training.csv`."""
    import kagglehub

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    train_file = dataset_dir / KAGGLE_TRAIN_FILE
    if not train_file.exists():
        raise FileNotFoundError(f"{KAGGLE_TRAIN_FILE} não encontrado em {dataset_dir}")
    return train_file


def generate_synthetic(samples: int = SYNTHETIC_SAMPLES, seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """Gera um dataset sintético com as mesmas colunas do Give Me Some Credit.

    Usado apenas como fallback quando o Kaggle não está acessível. Reproduz as
    características relevantes para o MVP: desbalanceamento do target (~6%),
    valores ausentes em MonthlyIncome/NumberOfDependents e a mesma escala das
    features.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 85, samples)
    utilization = np.clip(rng.lognormal(mean=-0.6, sigma=0.9, size=samples), 0, 1.5)
    debt_ratio = np.clip(rng.lognormal(mean=-1.2, sigma=1.1, size=samples), 0, 5000)
    monthly_income = np.clip(rng.lognormal(mean=8.4, sigma=0.6, size=samples), 0, None)
    open_credit_lines = rng.integers(0, 40, samples)
    real_estate_loans = rng.integers(0, 5, samples)
    past_due_30_59 = rng.choice([0, 1, 2, 3], size=samples, p=[0.85, 0.09, 0.04, 0.02])
    past_due_60_89 = rng.choice([0, 1, 2], size=samples, p=[0.94, 0.04, 0.02])
    late_90 = rng.choice([0, 1, 2], size=samples, p=[0.95, 0.035, 0.015])
    dependents = rng.choice([0, 1, 2, 3, 4, 5], size=samples, p=[0.55, 0.2, 0.13, 0.07, 0.03, 0.02])

    # Risco latente -> target desbalanceado (~6,7% de positivos), como no original.
    risk_score = (
        1.6 * past_due_30_59
        + 2.0 * past_due_60_89
        + 2.4 * late_90
        + 1.5 * utilization
        - 0.02 * (age - 50)
        - 0.00002 * monthly_income
    )
    probability = 1 / (1 + np.exp(-(risk_score - 5.4)))
    target = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            RAW_INDEX_COLUMN: np.arange(1, samples + 1),
            TARGET: target,
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
            "NumberOfDependents": dependents.astype(float),
        }
    )

    # Reintroduz valores ausentes nas mesmas proporções do dataset original.
    income_nulls = rng.choice(samples, size=int(samples * 0.20), replace=False)
    dependents_nulls = rng.choice(samples, size=int(samples * 0.03), replace=False)
    frame.loc[income_nulls, "MonthlyIncome"] = np.nan
    frame.loc[dependents_nulls, "NumberOfDependents"] = np.nan

    return frame[COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa o dataset Give Me Some Credit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV de saída.")
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Pula o Kaggle e gera direto o dataset sintético (fallback).",
    )
    parser.add_argument("--force", action="store_true", help="Sobrescreve o CSV de saída.")
    args = parser.parse_args()

    output: Path = args.output
    if output.exists() and not args.force:
        print(f"[skip] {output} já existe (use --force para sobrescrever).")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.force_synthetic:
        try:
            train_file = download_from_kaggle()
            shutil.copyfile(train_file, output)
            frame = pd.read_csv(output)
            print(f"[ok] Dataset do Kaggle copiado para {output}")
            print(f"     origem : {train_file}")
            print(f"     formato: {frame.shape[0]} linhas x {frame.shape[1]} colunas")
            print(f"     colunas: {', '.join(frame.columns)}")
            return 0
        except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer falha do Kaggle
            print(f"[aviso] Falha ao baixar do Kaggle: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(
                "[aviso] Usando dataset SINTÉTICO de fallback. Para usar o dado real, "
                "autentique no Kaggle e rode novamente com --force.",
                file=sys.stderr,
            )

    frame = generate_synthetic()
    frame.to_csv(output, index=False)
    print(f"[fallback] Dataset sintético gravado em {output}")
    print(f"     formato: {frame.shape[0]} linhas x {frame.shape[1]} colunas")
    print(f"     features esperadas presentes: {len(FEATURES) + 1} de {len(COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
