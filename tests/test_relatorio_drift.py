"""Etapa 2: relatório HTML de drift (Evidently) apontando as variáveis."""

from __future__ import annotations

import json
import re

import pytest
from conftest import construir_lote

from credito.drift import EXIT_DRIFT, EXIT_STABLE, comparar
from credito.esquema import FEATURES
from credito.relatorio_drift import gerar_html_drift


def _extrair_dashboard(html: str) -> dict:
    """Lê o JSON real do dashboard embutido no HTML (var metric_<hex>)."""
    match = re.search(r"var metric_[a-z0-9]+\s*=\s*", html)
    assert match is not None, "HTML não expõe a variável metric_<hex>"
    return json.JSONDecoder().raw_decode(html[match.end():])[0]


def _achatar(widgets: list) -> list:
    """Achata a árvore de widgets: grupos aninham a lista 'widgets'."""
    achados = []
    for widget in widgets:
        achados.append(widget)
        achados.extend(_achatar(widget.get("widgets") or []))
    return achados


def _contadores(widgets: list) -> list:
    """Junta os {label, value} de todos os widgets de contador."""
    itens = []
    for widget in widgets:
        if widget.get("type") == "counter":
            params = widget.get("params") or {}
            itens.extend(params.get("counters") or [])
    return itens


def _contador(itens: list, rotulo: str) -> dict:
    iguais = [item for item in itens if item["label"] == rotulo]
    assert len(iguais) == 1, f"contador {rotulo!r}: {len(iguais)} encontrados"
    return iguais[0]


def test_gera_html_de_drift(tmp_path):
    referencia = construir_lote(400, seed=1)
    producao = referencia.copy()
    producao["age"] += 20
    producao["MonthlyIncome"] *= 3
    destino = tmp_path / "reports" / "drift_report.html"

    gerar_html_drift(referencia, producao, destino)

    assert destino.exists()
    assert destino.stat().st_size > 0
    conteudo = destino.read_text(encoding="utf-8").lower()
    assert "<html" in conteudo
    assert "monthlyincome" in conteudo


def test_dashboard_html_explica_threshold_metodo_e_decisao(tmp_path):
    referencia = construir_lote(400)
    producao = referencia.copy()
    producao["age"] += 30

    reference_path = tmp_path / "referencia.csv"
    current_path = tmp_path / "producao.csv"
    report_path = tmp_path / "relatorio.json"
    html_path = tmp_path / "dashboard.html"
    referencia.to_csv(reference_path, index=False)
    producao.to_csv(current_path, index=False)

    comparar(
        reference_path,
        current_path,
        report_path,
        p_value=0.01,
        drift_share=0.1,
        html_path=html_path,
    )

    texto = html_path.read_text(encoding="utf-8")
    match = re.search(r"var metric_[a-z0-9]+\s*=\s*", texto)
    assert match is not None, "HTML não expõe a variável metric_<hex>"
    dashboard = json.JSONDecoder().raw_decode(texto[match.end():])[0]
    serializado = json.dumps(dashboard)

    assert "Dataset drift detection threshold is 0.1" in serializado
    assert "chi-square" not in serializado
    assert "Dataset Drift is detected." in serializado

    relatorio = json.loads(report_path.read_text(encoding="utf-8"))
    assert relatorio["status"] == "drift_detectado"


@pytest.mark.parametrize(
    ("deslocamento_age", "share", "esperado"),
    [
        pytest.param(30, 0.1, True, id="idade+30_share_0.1"),
        pytest.param(30, 0.2, False, id="idade+30_share_0.2"),
        pytest.param(0, 0.1, False, id="lote_identico"),
    ],
)
def test_dashboard_e_json_concordam(tmp_path, deslocamento_age, share, esperado):
    """Tabela, contadores e status do HTML vêm dos mesmos dados do JSON."""
    referencia = construir_lote(400)
    producao = referencia.copy()
    producao["age"] += deslocamento_age

    reference_path = tmp_path / "referencia.csv"
    current_path = tmp_path / "producao.csv"
    report_path = tmp_path / "relatorio.json"
    html_path = tmp_path / "dashboard.html"
    referencia.to_csv(reference_path, index=False)
    producao.to_csv(current_path, index=False)

    codigo = comparar(
        reference_path,
        current_path,
        report_path,
        p_value=0.01,
        drift_share=share,
        html_path=html_path,
    )

    dashboard = _extrair_dashboard(html_path.read_text(encoding="utf-8"))
    widgets = _achatar(dashboard["widgets"])
    relatorio = json.loads(report_path.read_text(encoding="utf-8"))
    agregado = relatorio["agregado"]

    # big_table: uma linha por feature com o teste aplicado.
    tabelas = [
        widget
        for widget in widgets
        if widget.get("type") == "big_table"
        and isinstance(widget.get("params"), dict)
        and widget["params"].get("data")
    ]
    assert len(tabelas) == 1, "HTML não tem a tabela de drift por coluna"
    brutas = tabelas[0]["params"]["data"]
    linhas = {
        linha["column_name"]: linha for linha in brutas if "column_name" in linha
    }
    assert len(linhas) == len(brutas), "linha da tabela sem column_name"
    assert set(linhas) == set(FEATURES) == set(relatorio["colunas"])

    obrigatorias = {"column_name", "stattest_name", "data_drift", "drift_score"}
    for nome, info in relatorio["colunas"].items():
        linha = linhas[nome]
        assert obrigatorias <= set(linha)
        assert linha["stattest_name"] == "K-S p_value"
        drift_esperado = "Detected" if info["drift"] else "Not Detected"
        assert linha["data_drift"] == drift_esperado
        # O HTML guarda o score arredondado em 6 casas: 1e-6 cobre o corte.
        assert linha["drift_score"] == pytest.approx(info["p_value"], abs=1e-6)

    detectadas = sum(
        1 for linha in linhas.values() if linha["data_drift"] == "Detected"
    )
    assert detectadas == agregado["colunas_com_drift"]

    itens = _contadores(widgets)
    total_html = _contador(itens, "Columns")["value"]
    com_drift_html = _contador(itens, "Drifted Columns")["value"]
    share_html = _contador(itens, "Share of Drifted Columns")["value"]
    assert total_html == str(agregado["total_colunas"])
    assert com_drift_html == str(agregado["colunas_com_drift"])
    assert float(share_html) == pytest.approx(agregado["share"], abs=1e-3)

    titulos = [
        item["label"]
        for item in itens
        if item["label"].startswith("Dataset Drift is")
    ]
    assert len(titulos) == 1, "HTML não tem o contador de status do dataset"
    esperado_titulo = (
        "Dataset Drift is detected."
        if esperado
        else "Dataset Drift is NOT detected."
    )
    assert esperado_titulo in titulos[0]
    assert f"Dataset drift detection threshold is {share}" in titulos[0]
    assert agregado["drift"] is esperado
    assert relatorio["status"] == ("drift_detectado" if esperado else "sem_drift")
    assert codigo == (EXIT_DRIFT if esperado else EXIT_STABLE)

    configuracao = relatorio["configuracao"]
    assert configuracao["p_value_limite"] == 0.01
    assert configuracao["share_minimo"] == share
    assert configuracao["metodo"] == "ks"
