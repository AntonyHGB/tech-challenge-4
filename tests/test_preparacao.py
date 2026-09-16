"""Testes da limpeza do dataset de referência."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credito.esquema import COLUMNS, PAST_DUE_COLUMNS, RAW_INDEX_COLUMN
from credito.preparacao import preparar_referencia


def _lote_bruto() -> pd.DataFrame:
    """Lote com índice do Kaggle, nulos, duplicata, idade inválida e sentinela."""
    linhas = [
        (0, 0.10, 30, 0, 0.20, 5000.0, 5, 0, 1, 0, 1.0),
        (1, 0.90, 45, 2, 1.50, 3000.0, 8, 0, 0, 0, 2.0),
        (0, 0.20, 17, 0, 0.10, 4000.0, 4, 0, 0, 0, 0.0),  # idade <= 18
        (0, 0.30, 50, 0, 0.50, np.nan, 6, 0, 1, 0, 1.0),  # renda nula
        (1, 0.40, 60, 1, 0.30, 7000.0, 7, 0, 2, 0, np.nan),  # dependentes nulos
        (1, 0.90, 45, 2, 1.50, 3000.0, 8, 0, 0, 0, 2.0),  # duplicata exata
        (0, 0.50, 35, 96, 0.40, 6000.0, 6, 0, 1, 0, 3.0),  # sentinela 96
    ]
    frame = pd.DataFrame(linhas, columns=COLUMNS)
    frame.insert(0, RAW_INDEX_COLUMN, range(1, len(frame) + 1))
    return frame


def test_preparacao_remove_nulos_duplicatas_e_defeitos(tmp_path):
    """Sobrevivem apenas as duas linhas válidas do lote bruto."""
    entrada = tmp_path / "bruto.csv"
    saida = tmp_path / "referencia.csv"
    _lote_bruto().to_csv(entrada, index=False)

    preparada = preparar_referencia(entrada, saida)

    assert RAW_INDEX_COLUMN not in preparada.columns
    assert list(preparada.columns) == COLUMNS
    assert len(preparada) == 2
    assert not preparada.isna().any().any()
    assert not preparada.duplicated().any()
    assert (preparada["age"] > 18).all()
    assert int(preparada[PAST_DUE_COLUMNS].to_numpy().max()) <= 50
    assert list(pd.read_csv(saida).columns) == COLUMNS
    assert len(pd.read_csv(saida)) == 2


def test_preparacao_exige_arquivo_de_entrada(tmp_path):
    """Entrada ausente falha com FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preparar_referencia(tmp_path / "ausente.csv", tmp_path / "saida.csv")
