"""Smoke do treino do baseline com dataset minúsculo gravado em tmp_path."""

from __future__ import annotations

import json

import joblib

from credito.modelo import construir_modelo, treinar

METRICAS = ("roc_auc", "accuracy", "precision", "recall", "f1")


def test_pipeline_tem_imputacao_padronizacao_e_classificador():
    modelo = construir_modelo()
    assert [nome for nome, _ in modelo.steps] == ["imputer", "scaler", "classifier"]


def test_treino_grava_modelo_e_metricas(lote_limpo, tmp_path):
    """Treinar com um lote sintético minúsculo produz artefatos e métricas válidas."""
    entrada = tmp_path / "referencia.csv"
    modelo_path = tmp_path / "modelo.joblib"
    metricas_path = tmp_path / "baseline_metrics.json"
    lote_limpo.to_csv(entrada, index=False)

    metricas = treinar(entrada, modelo_path, metricas_path)

    assert modelo_path.exists()
    assert metricas_path.exists()
    assert joblib.load(modelo_path) is not None
    assert metricas["n_total"] == len(lote_limpo)
    assert metricas["n_train"] + metricas["n_test"] == len(lote_limpo)
    for nome in METRICAS:
        assert 0 <= metricas[nome] <= 1
    assert json.loads(metricas_path.read_text()) == metricas
