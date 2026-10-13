"""Etapa 2: Dataset de Produção simulado — determinístico e dentro do contrato."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from conftest import construir_lote

from credito.contrato_dados import validar_dataframe
from credito.esquema import FEATURES
from credito.simulacao_producao import (
    ESTAGIOS,
    FEATURES_DESLOCADAS,
    FEATURES_ESTAVEIS,
    main,
    simular_producao,
)


def test_simulacao_e_deterministica():
    referencia = construir_lote(200)
    primeira = simular_producao(referencia, seed=13, estagios=3)
    segunda = simular_producao(referencia, seed=13, estagios=3)
    assert [nome for nome, _ in primeira] == [
        "producao_t1.csv",
        "producao_t2.csv",
        "producao_t3.csv",
    ]
    for (_, a), (_, b) in zip(primeira, segunda, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_semente_diferente_muda_o_lote():
    referencia = construir_lote(200)
    a = simular_producao(referencia, seed=1, estagios=1)[0][1]
    b = simular_producao(referencia, seed=2, estagios=1)[0][1]
    assert not a["MonthlyIncome"].equals(b["MonthlyIncome"])


def test_ao_menos_duas_features_sao_deslocadas():
    referencia = construir_lote(400)
    producao = simular_producao(referencia, seed=7, estagios=ESTAGIOS)[-1][1]
    deslocadas = [
        coluna
        for coluna in FEATURES
        if not producao[coluna].equals(referencia[coluna])
    ]
    assert len(deslocadas) >= 2
    assert set(FEATURES_DESLOCADAS).issubset(deslocadas)


def test_features_estaveis_permanecem_iguais():
    referencia = construir_lote(400)
    producao = simular_producao(referencia, seed=7, estagios=ESTAGIOS)[-1][1]
    for coluna in FEATURES_ESTAVEIS:
        pd.testing.assert_series_equal(producao[coluna], referencia[coluna])


def test_dataset_de_producao_passa_no_contrato():
    """Critério de aceite: o Dataset de Produção é DRIFT, não corrupção."""
    referencia = construir_lote(400)
    for _, estagio in simular_producao(referencia, seed=7, estagios=ESTAGIOS):
        resultado = validar_dataframe(estagio)
        violadas = [
            detail.expectation_config.meta["rule"]
            for detail in resultado.results
            if not detail.success
        ]
        assert resultado.success, f"contrato violado: {violadas}"


def test_estagios_progridem_o_deslocamento():
    referencia = construir_lote(400)
    estagios = simular_producao(referencia, seed=7, estagios=3)
    rendas = [frame["MonthlyIncome"].mean() for _, frame in estagios]
    idades = [frame["age"].mean() for _, frame in estagios]
    assert rendas[0] < rendas[1] < rendas[2]
    assert idades[0] < idades[1] < idades[2]


def test_estagios_invalidos_reprovam():
    with pytest.raises(ValueError):
        simular_producao(construir_lote(60), seed=1, estagios=0)


def test_cli_grava_um_csv_por_estagio(tmp_path, monkeypatch):
    entrada = tmp_path / "reference.csv"
    construir_lote(120).to_csv(entrada, index=False)
    destino = tmp_path / "production"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simular_producao.py",
            "--input",
            str(entrada),
            "--output",
            str(destino),
            "--seed",
            "5",
            "--stages",
            "2",
        ],
    )

    assert main() == 0
    assert (destino / "producao_t1.csv").exists()
    assert (destino / "producao_t2.csv").exists()
    assert not (destino / "producao_t3.csv").exists()
