"""Etapa 2: predições do baseline para o Dataset de Produção."""

from __future__ import annotations

import sys

import joblib
import pandas as pd
import pytest
from conftest import construir_lote

from credito.esquema import FEATURES, TARGET
from credito.modelo import construir_modelo
from credito.predicao import CLASSE, PROBABILIDADE, carregar_modelo, main, predizer


def _treinar_modelo(destino, lote):
    modelo = construir_modelo()
    modelo.fit(lote[FEATURES], lote[TARGET])
    joblib.dump(modelo, destino)
    return destino


def test_modelo_ausente_falha_claramente(tmp_path, monkeypatch, capsys):
    entrada = tmp_path / "producao.csv"
    construir_lote(60).to_csv(entrada, index=False)
    falta = tmp_path / "sem_modelo.joblib"

    with pytest.raises(FileNotFoundError):
        carregar_modelo(falta)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predizer.py",
            "--input",
            str(entrada),
            "--model",
            str(falta),
            "--output",
            str(tmp_path / "predicoes.csv"),
        ],
    )
    assert main() == 2
    assert "não encontrado" in capsys.readouterr().err


def test_predicao_grava_probabilidade_e_classe(tmp_path):
    lote = construir_lote(120)
    entrada = tmp_path / "producao.csv"
    lote.to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", lote)
    saida = tmp_path / "predicoes.csv"

    metricas = predizer(entrada, modelo_path, saida)

    resultado = pd.read_csv(saida)
    assert len(resultado) == len(lote)
    assert PROBABILIDADE in resultado.columns
    assert CLASSE in resultado.columns
    assert resultado[PROBABILIDADE].between(0, 1).all()
    assert set(resultado[CLASSE].unique()).issubset({0, 1})
    assert metricas["n"] == len(lote)
    for nome in ("roc_auc", "accuracy", "precision", "recall", "f1"):
        assert 0 <= metricas[nome] <= 1


def test_predicao_sem_rotulo_nao_calcula_metricas(tmp_path):
    lote = construir_lote(120)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", lote)
    entrada = tmp_path / "producao_sem_rotulo.csv"
    lote.drop(columns=[TARGET]).to_csv(entrada, index=False)
    saida = tmp_path / "predicoes.csv"

    assert predizer(entrada, modelo_path, saida) is None

    resultado = pd.read_csv(saida)
    assert PROBABILIDADE in resultado.columns
    assert TARGET not in resultado.columns
