"""Etapa 2: detecção pontual de drift de features, após validação do contrato."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from credito.contrato_dados import validar_dataframe
from credito.esquema import FEATURES

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_REPORT = RAIZ / "artifacts" / "drift_report.json"
MIN_ROWS = 100
P_VALUE = 0.05
DRIFT_SHARE = 0.5

# Mesmo padrão do validador: 0 aprovado, 1 contrato reprovado, 2 erro.
# 3 distingue drift estatístico de lote inválido.
EXIT_STABLE = 0
EXIT_INVALID = 1
EXIT_ERROR = 2
EXIT_DRIFT = 3


def _identidade(path: Path, frame: pd.DataFrame) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "arquivo": _caminho_relatorio(path),
        "sha256": digest.hexdigest(),
        "linhas": len(frame),
    }


def _caminho_relatorio(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        # Fora do diretório de trabalho: não revelar a árvore de diretórios.
        return resolved.name


def _gravar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def comparar(
    reference_path: Path,
    current_path: Path,
    report_path: Path,
    min_rows: int = MIN_ROWS,
    p_value: float = P_VALUE,
    drift_share: float = DRIFT_SHARE,
) -> int:
    """Escreve relatório e retorna 0=estável, 1=inválido, 2=erro, 3=drift."""
    if report_path.resolve() in (reference_path.resolve(), current_path.resolve()):
        raise ValueError(
            "O caminho do relatório deve ser diferente dos CSVs de entrada"
        )
    # Uma falha posterior não deve deixar um resultado de execução anterior utilizável.
    report_path.unlink(missing_ok=True)
    if min_rows < 2 or not 0 < p_value < 1 or not 0 < drift_share <= 1:
        raise ValueError("Use min_rows >= 2, 0 < p_value < 1 e 0 < drift_share <= 1")
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    payload = {
        "referencia": _identidade(reference_path, reference),
        "lote": _identidade(current_path, current),
        "configuracao": {
            "biblioteca": "evidently==0.7.23",
            "features": FEATURES,
            "metodo": "ks",
            "p_value_limite": p_value,
            "share_minimo": drift_share,
            "min_rows": min_rows,
            "regra_coluna": "p_value < p_value_limite",
            "regra_agregada": "share >= share_minimo",
        },
    }

    # A referência inválida é erro de configuração, não lote inválido.
    reference_result = validar_dataframe(reference)
    if not reference_result.success:
        raise ValueError("Referência não passou no contrato de dados")

    current_result = validar_dataframe(current)
    if not current_result.success:
        payload["status"] = "lote_invalido"
        payload["regras_violadas"] = sorted(
            result.expectation_config.meta["rule"]
            for result in current_result.results
            if not result.success
        )
        _gravar(report_path, payload)
        return EXIT_INVALID

    if len(reference) < min_rows or len(current) < min_rows:
        payload["status"] = "amostra_insuficiente"
        _gravar(report_path, payload)
        return EXIT_ERROR

    definition = DataDefinition(numerical_columns=FEATURES)
    ref_data = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    cur_data = Dataset.from_pandas(current[FEATURES], data_definition=definition)
    metrics = [
        ValueDrift(column=column, method="ks", threshold=p_value) for column in FEATURES
    ]
    snapshot = Report(metrics).run(cur_data, ref_data)
    columns = {}
    for metric in snapshot.dict()["metrics"]:
        column = metric["config"]["column"]
        score = float(metric["value"])
        columns[column] = {"p_value": score, "drift": score < p_value}
    if set(columns) != set(FEATURES):
        raise ValueError("Evidently não retornou resultado para todas as features")
    count = sum(info["drift"] for info in columns.values())
    share = count / len(FEATURES)
    detected = share >= drift_share
    payload.update(
        {
            "status": "drift_detectado" if detected else "sem_drift",
            "colunas": columns,
            "agregado": {
                "colunas_com_drift": count,
                "total_colunas": len(FEATURES),
                "share": share,
                "drift": detected,
            },
        }
    )
    _gravar(report_path, payload)
    return EXIT_DRIFT if detected else EXIT_STABLE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida lote e detecta drift de features."
    )
    parser.add_argument("lote", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--p-value", type=float, default=P_VALUE)
    parser.add_argument("--drift-share", type=float, default=DRIFT_SHARE)
    args = parser.parse_args()
    try:
        code = comparar(
            args.reference,
            args.lote,
            args.report,
            args.min_rows,
            args.p_value,
            args.drift_share,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    labels = {
        EXIT_STABLE: "sem_drift",
        EXIT_INVALID: "lote_invalido",
        EXIT_ERROR: "amostra_insuficiente",
        EXIT_DRIFT: "drift_detectado",
    }
    print(f"RESULTADO: {labels[code]} (exit code {code}); relatório: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
