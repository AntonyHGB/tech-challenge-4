"""Etapa 2: Dataset de Produção simulado — determinístico e dentro do contrato."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from conftest import construir_lote

from credito.contrato_dados import validar_dataframe
from credito.esquema import FEATURES, TARGET
from credito.modelo import dividir_referencia
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


def test_concept_drift_aumenta_a_inadimplencia_real():
    """Os rótulos mudam (0 -> 1) e a taxa de inadimplência sobe a cada estágio."""
    referencia = construir_lote(2000)
    taxas = [
        frame[TARGET].mean()
        for _, frame in simular_producao(referencia, seed=7, estagios=3)
    ]
    assert referencia[TARGET].mean() < taxas[0] < taxas[1] < taxas[2]


def test_concept_drift_so_vira_rotulo_de_0_para_1():
    referencia = construir_lote(2000)
    producao = simular_producao(referencia, seed=7, estagios=3)[-1][1]
    mudou = producao[TARGET] != referencia[TARGET]
    assert mudou.any()
    assert (referencia.loc[mudou, TARGET] == 0).all()
    assert (producao.loc[mudou, TARGET] == 1).all()


def test_fracao_atingida_pela_crise_cresce():
    """Quanto mais tempo passa, mais clientes têm atraso acrescentado."""
    referencia = construir_lote(2000)
    coluna = "NumberOfTimes90DaysLate"
    fracoes = [
        (frame[coluna] > referencia[coluna]).mean()
        for _, frame in simular_producao(referencia, seed=7, estagios=3)
    ]
    assert fracoes[0] < fracoes[1] < fracoes[2] < 1.0


def test_cli_usa_so_o_holdout(tmp_path, monkeypatch):
    """A produção sai do holdout (20%), nunca das linhas usadas no treino."""
    entrada = tmp_path / "reference.csv"
    referencia = construir_lote(500)
    referencia.to_csv(entrada, index=False)
    destino = tmp_path / "production"
    monkeypatch.setattr(
        sys,
        "argv",
        ["simular_producao.py", "--input", str(entrada), "--output", str(destino)],
    )

    assert main() == 0
    producao = pd.read_csv(destino / "producao_t1.csv")
    _, holdout = dividir_referencia(referencia)
    assert len(producao) == len(holdout)
    # Features que a simulação não mexe batem com as do holdout, linha a linha.
    pd.testing.assert_frame_equal(
        producao[FEATURES_ESTAVEIS],
        holdout[FEATURES_ESTAVEIS].reset_index(drop=True),
    )
