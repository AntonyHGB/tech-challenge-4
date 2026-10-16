"""Etapa 2: predições do baseline para o Dataset de Produção."""

from __future__ import annotations

import sys
import warnings

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


def _bytes_previos(*caminhos):
    return {caminho: caminho.read_bytes() for caminho in caminhos}


def test_features_ausentes_mantem_erro_claro(tmp_path):
    """A mensagem de features faltantes continua explícita e nada é gravado."""
    entrada = tmp_path / "producao_sem_idade.csv"
    construir_lote(60).drop(columns=["age"]).to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(60))
    saida = tmp_path / "predicoes.csv"

    with pytest.raises(ValueError, match="Colunas de feature ausentes no CSV"):
        predizer(entrada, modelo_path, saida)

    assert not saida.exists()


def _aplicar_defeito(frame, defeito):
    frame = frame.copy()
    if defeito == "idade_menor":
        frame.loc[0, "age"] = 16
    elif defeito == "renda_negativa":
        frame.loc[1, "MonthlyIncome"] = -500.0
    elif defeito == "renda_nula":
        frame.loc[1, "MonthlyIncome"] = float("nan")
    elif defeito == "duplicata":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:  # pragma: no cover - defeito de teste desconhecido
        raise AssertionError(f"defeito desconhecido: {defeito}")
    return frame


@pytest.mark.parametrize(
    "defeito",
    ["idade_menor", "renda_negativa", "renda_nula", "duplicata"],
)
def test_lote_invalido_no_contrato_bloqueia_saida(tmp_path, defeito):
    """Lote reprovado no contrato não grava CSV nem toca nos artefatos."""
    lote = _aplicar_defeito(construir_lote(120), defeito)
    entrada = tmp_path / "producao.csv"
    lote.to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(120))
    saida = tmp_path / "predicoes.csv"
    previos = _bytes_previos(entrada, modelo_path)

    with pytest.raises(ValueError, match="contrato de dados"):
        predizer(entrada, modelo_path, saida)

    assert not saida.exists()
    assert _bytes_previos(entrada, modelo_path) == previos


@pytest.mark.parametrize("rotulo_ruim", [2, 0.5, float("nan")])
def test_target_invalido_falha_antes_de_gravar(tmp_path, rotulo_ruim):
    """Rótulo não binário/nulo reprova antes de qualquer escrita."""
    lote = construir_lote(120)
    lote[TARGET] = lote[TARGET].astype(float)
    lote.loc[0, TARGET] = rotulo_ruim
    entrada = tmp_path / "producao.csv"
    lote.to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(120))
    saida = tmp_path / "predicoes.csv"
    previos = _bytes_previos(entrada, modelo_path)

    with pytest.raises(ValueError, match=TARGET):
        predizer(entrada, modelo_path, saida)

    assert not saida.exists()
    assert _bytes_previos(entrada, modelo_path) == previos


def test_lote_com_uma_classe_calcula_metricas_com_roc_auc_nulo(tmp_path, capsys):
    """Rótulo de uma única classe: demais métricas saem, roc_auc vira N/A sem nan."""
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(120))
    lote = construir_lote(80, seed=7)
    lote[TARGET] = 0
    entrada = tmp_path / "producao_uma_classe.csv"
    lote.to_csv(entrada, index=False)
    saida = tmp_path / "predicoes.csv"

    with warnings.catch_warnings(record=True) as alertas:
        warnings.simplefilter("always")
        metricas = predizer(entrada, modelo_path, saida)

    assert metricas is not None
    assert metricas["roc_auc"] is None
    assert metricas["n"] == len(lote)
    for nome in ("accuracy", "precision", "recall", "f1"):
        assert 0 <= metricas[nome] <= 1

    resultado = pd.read_csv(saida)
    assert PROBABILIDADE in resultado.columns

    texto = capsys.readouterr().out
    assert "roc_auc=N/A" in texto
    assert "nan" not in texto.lower()
    relacionados = [
        str(item.message)
        for item in alertas
        if "roc" in str(item.message).lower()
        or "ill-defined" in str(item.message).lower()
    ]
    assert relacionados == []


def test_saida_nao_sobrescreve_input(tmp_path):
    """--output igual ao input é recusado e o CSV de entrada fica intacto."""
    entrada = tmp_path / "producao.csv"
    construir_lote(60).to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(60))
    previos = _bytes_previos(entrada, modelo_path)

    with pytest.raises(ValueError, match="sobrescrever"):
        predizer(entrada, modelo_path, entrada)

    assert _bytes_previos(entrada, modelo_path) == previos


def test_saida_nao_sobrescreve_modelo(tmp_path):
    """--output igual ao modelo é recusado e o artefato joblib fica intacto."""
    entrada = tmp_path / "producao.csv"
    construir_lote(60).to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(60))
    previos = _bytes_previos(entrada, modelo_path)

    with pytest.raises(ValueError, match="sobrescrever"):
        predizer(entrada, modelo_path, modelo_path)

    assert _bytes_previos(entrada, modelo_path) == previos


def test_main_retorna_2_quando_lote_reprovado(tmp_path, monkeypatch, capsys):
    """A CLI reporta o erro no stderr (exit 2) sem traceback."""
    lote = _aplicar_defeito(construir_lote(60), "idade_menor")
    entrada = tmp_path / "producao.csv"
    lote.to_csv(entrada, index=False)
    modelo_path = _treinar_modelo(tmp_path / "modelo.joblib", construir_lote(60))
    saida = tmp_path / "predicoes.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predizer.py",
            "--input",
            str(entrada),
            "--model",
            str(modelo_path),
            "--output",
            str(saida),
        ],
    )

    assert main() == 2
    saida_erro = capsys.readouterr().err
    assert "[erro]" in saida_erro
    assert "Traceback" not in saida_erro
    assert not saida.exists()
