"""Testes do gerador sintético de fallback e do script de aquisição."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from credito.aquisicao import gerar_sintetico
from credito.esquema import COLUMNS, PAST_DUE_COLUMNS, TARGET

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "baixar_dataset.py"


def test_gerador_sintetico_segue_o_esquema_do_dataset():
    """O fallback tem as mesmas colunas, target desbalanceado e nulos de propósito."""
    frame = gerar_sintetico(samples=300, seed=7)

    assert list(frame.columns) == COLUMNS
    assert len(frame) == 300
    assert set(frame[TARGET]) <= {0, 1}
    assert 0 < frame[TARGET].mean() < 0.2
    assert (frame["age"] > 18).all()
    assert int(frame[PAST_DUE_COLUMNS].to_numpy().max()) <= 50
    assert frame["MonthlyIncome"].isna().any()


def test_gerador_sintetico_e_deterministico():
    """Mesma seed produz exatamente o mesmo dataset."""
    primeira = gerar_sintetico(samples=50, seed=1)
    segunda = gerar_sintetico(samples=50, seed=1)
    assert primeira.equals(segunda)


def test_script_de_aquisicao_usa_fallback_sintetico(tmp_path):
    """--force-synthetic grava o CSV sem tentar acessar o Kaggle."""
    destino = tmp_path / "raw" / "dataset.csv"

    processo = subprocess.run(
        [sys.executable, str(SCRIPT), "--force-synthetic", "--output", str(destino)],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )

    assert processo.returncode == 0
    assert destino.exists()
    assert list(pd.read_csv(destino).columns) == COLUMNS
    assert "sintético" in processo.stdout
