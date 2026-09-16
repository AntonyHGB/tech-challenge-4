"""Fixtures com lotes sintéticos pequenos — a suíte não acessa rede nem Kaggle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.esquema import COLUMNS, TARGET

LINHAS = 60
SEED = 42


def construir_lote(linhas: int = LINHAS, seed: int = SEED) -> pd.DataFrame:
    """Monta um lote limpo e minúsculo, válido para as 13 regras do contrato."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            TARGET: rng.integers(0, 2, size=linhas),
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, size=linhas),
            "age": rng.integers(21, 75, size=linhas),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 4, size=linhas),
            "DebtRatio": rng.uniform(0, 2, size=linhas).round(4),
            "MonthlyIncome": rng.uniform(1000, 9000, size=linhas).round(2),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, size=linhas),
            "NumberOfTimes90DaysLate": rng.integers(0, 3, size=linhas),
            "NumberRealEstateLoansOrLines": rng.integers(0, 4, size=linhas),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, size=linhas),
            "NumberOfDependents": rng.integers(0, 5, size=linhas).astype(float),
        }
    )
    frame = frame[COLUMNS]
    assert not frame.duplicated().any(), (
        "fixture precisa partir de um lote sem duplicatas"
    )
    return frame


@pytest.fixture
def lote_limpo() -> pd.DataFrame:
    """Lote pequeno sem defeitos: deve ser aprovado pelo contrato."""
    return construir_lote()
