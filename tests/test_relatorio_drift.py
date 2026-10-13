"""Etapa 2: relatório HTML de drift (Evidently) apontando as variáveis."""

from __future__ import annotations

from conftest import construir_lote

from credito.relatorio_drift import gerar_html_drift


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
