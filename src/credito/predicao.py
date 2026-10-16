"""Etapa 2 (2/3) — Predições do baseline sobre o Dataset de Produção.

Carrega o modelo treinado em `artifacts/model.joblib` (falha com mensagem clara
se ele não existir) e o aplica a um dataset de produção, gravando um CSV com a
`probabilidade_inadimplencia` e a `classe_prevista`. Se o rótulo
(`SeriousDlqin2yrs`) estiver presente, também reporta métricas no console.

Uso:
    python scripts/predizer.py
    python scripts/predizer.py --input data/production/producao_t3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credito.contrato_dados import PERFIL_INFERENCIA, validar_dataframe
from credito.esquema import FEATURES, TARGET, VALID_TARGET_VALUES

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = RAIZ / "artifacts" / "model.joblib"
DEFAULT_INPUT = RAIZ / "data" / "production" / "producao_t1.csv"
DEFAULT_OUTPUT = RAIZ / "data" / "production" / "predicoes_t1.csv"

PROBABILIDADE = "probabilidade_inadimplencia"
CLASSE = "classe_prevista"
LIMITE = 0.5


def carregar_modelo(model_path: Path):
    """Carrega o modelo joblib; erro claro se o artefato não existir."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            "Rode antes: python scripts/treinar_modelo.py"
        )
    return joblib.load(model_path)


def _bloquear_saida_destrutiva(
    output_csv: Path, input_csv: Path, model_path: Path
) -> None:
    """Recusa --output igual ao input ou ao modelo (antes de ler/carregar/gravar)."""
    saida = output_csv.resolve()
    if saida == input_csv.resolve():
        alvo = "o dataset de entrada"
    elif saida == model_path.resolve():
        alvo = "o modelo"
    else:
        return
    raise ValueError(
        f"Saída não pode sobrescrever {alvo}: --output={output_csv}. "
        "Escolha outro caminho de saída."
    )


def _nome_da_regra(expectation_config) -> str:
    meta = getattr(expectation_config, "meta", None) or {}
    return meta.get("rule") or expectation_config.type


def _validar_lote(frame: pd.DataFrame) -> None:
    """Aplica o contrato no perfil de inferência antes de carregar/aplicar o modelo."""
    resultado = validar_dataframe(frame, PERFIL_INFERENCIA)
    if resultado.success:
        return
    violadas = sorted(
        _nome_da_regra(detalle.expectation_config)
        for detalle in resultado.results
        if not detalle.success
    )
    raise ValueError(
        "Dataset de produção não passou no contrato de dados "
        f"(perfil {PERFIL_INFERENCIA}); regras violadas: {', '.join(violadas)}"
    )


def _rotulo_valido(frame: pd.DataFrame) -> pd.Series | None:
    """Rótulo ausente -> None; presente precisa ser finito e binário (0/1)."""
    if TARGET not in frame.columns:
        return None
    rotulo = pd.to_numeric(frame[TARGET], errors="coerce")
    if rotulo.isna().any() or not np.isfinite(rotulo).all():
        raise ValueError(
            f"Rótulo '{TARGET}' contém valores nulos ou não finitos; "
            "corrija o dataset antes de gravar as predições."
        )
    if not rotulo.isin(VALID_TARGET_VALUES).all():
        distintos = sorted(set(rotulo.tolist()))[:5]
        raise ValueError(
            f"Rótulo '{TARGET}' deve ser binário {VALID_TARGET_VALUES}; "
            f"valores encontrados (amostra): {distintos}"
        )
    return rotulo


def _calcular_metricas(
    rotulo: pd.Series, probabilidades, classes
) -> dict[str, float | int | None]:
    """Métricas do lote; `roc_auc` fica None quando o rótulo tem uma só classe."""
    com_roc = rotulo.nunique() >= 2
    return {
        "n": int(len(rotulo)),
        "taxa_prevista_positivos": round(float(classes.mean()), 4),
        "roc_auc": (
            round(float(roc_auc_score(rotulo, probabilidades)), 4) if com_roc else None
        ),
        "accuracy": round(float(accuracy_score(rotulo, classes)), 4),
        "precision": round(float(precision_score(rotulo, classes, zero_division=0)), 4),
        "recall": round(float(recall_score(rotulo, classes, zero_division=0)), 4),
        "f1": round(float(f1_score(rotulo, classes, zero_division=0)), 4),
    }


def predizer(input_csv: Path, model_path: Path, output_csv: Path) -> dict | None:
    """Gera prob/classe para o CSV e devolve as métricas (ou None sem rótulo)."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Dataset de produção não encontrado: {input_csv}. "
            "Rode antes: python scripts/simular_producao.py"
        )

    _bloquear_saida_destrutiva(output_csv, input_csv, model_path)

    frame = pd.read_csv(input_csv)

    faltantes = [coluna for coluna in FEATURES if coluna not in frame.columns]
    if faltantes:
        raise ValueError(f"Colunas de feature ausentes no CSV: {faltantes}")

    _validar_lote(frame)
    rotulo = _rotulo_valido(frame)

    model = carregar_modelo(model_path)

    probabilidades = model.predict_proba(frame[FEATURES])[:, 1]
    classes = (probabilidades >= LIMITE).astype(int)

    # Métricas antes de qualquer escrita: rótulo inválido nunca gera CSV parcial.
    metricas = (
        _calcular_metricas(rotulo, probabilidades, classes)
        if rotulo is not None
        else None
    )

    resultado = frame.copy()
    resultado[PROBABILIDADE] = probabilidades.round(6)
    resultado[CLASSE] = classes

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_csv, index=False)

    print("Predição do dataset de produção concluída:")
    print(f"  - modelo : {model_path}")
    print(f"  - entrada: {input_csv} ({len(frame)} linhas)")
    print(f"  - saída  : {output_csv}")
    print(f"  - colunas: '{PROBABILIDADE}', '{CLASSE}'")
    if metricas is not None:
        taxa = metricas["taxa_prevista_positivos"]
        roc = metricas["roc_auc"]
        roc_texto = "N/A (rótulo com uma só classe)" if roc is None else f"{roc:.4f}"
        print(f"  - taxa prevista (classe=1): {taxa:.4f}")
        print(
            "  - métricas (rótulo presente): "
            f"roc_auc={roc_texto}, "
            f"accuracy={metricas['accuracy']:.4f}, "
            f"precision={metricas['precision']:.4f}, "
            f"recall={metricas['recall']:.4f}, "
            f"f1={metricas['f1']:.4f}"
        )
    else:
        print("  - sem rótulo: métricas não calculadas")
    return metricas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera predições do baseline para um dataset de produção."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        predizer(args.input, args.model, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
