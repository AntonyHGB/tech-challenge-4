"""Etapa 2 (3/3) — Relatório HTML de drift do Evidently.

Gera o entregável visual (HTML) exigido pelo enunciado, apontando as variáveis
com drift entre a Referência (treino) e o Dataset de Produção. Complementa —
não substitui — o relatório JSON e os exit codes de `credito/drift.py`.

Uso (indireto, via `scripts/detectar_drift.py`):
    from credito.relatorio_drift import gerar_html_drift
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from credito.esquema import FEATURES


def gerar_html_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    p_value: float = 0.05,
    drift_share: float = 0.5,
) -> Path:
    """Compara Referência x Produção e grava o relatório HTML do Evidently."""
    definition = DataDefinition(numerical_columns=FEATURES)
    ref_data = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    cur_data = Dataset.from_pandas(current[FEATURES], data_definition=definition)

    preset = DataDriftPreset(
        columns=FEATURES,
        method="ks",
        threshold=p_value,
        drift_share=drift_share,
    )
    snapshot = Report([preset]).run(cur_data, ref_data)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    return html_path
