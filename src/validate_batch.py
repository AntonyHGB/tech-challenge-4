"""Etapa 1 (4/5) — Validação de lote com bloqueio de ingestão.

Valida um CSV contra o contrato de dados (`src/data_contract.py`) e usa o
exit code para simular o bloqueio da pipeline de ingestão:

    0 -> lote aprovado (ingestão liberada)
    1 -> lote reprovado (ingestão BLOQUEADA)
    2 -> erro de execução (arquivo ausente, CSV inválido, etc.)

Uso:
    python src/validate_batch.py data/reference/reference.csv
    python src/validate_batch.py data/corrupted/corrupted_batch.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_contract import SUITE_NAME, validate_dataframe

EXIT_OK = 0
EXIT_BLOCKED = 1
EXIT_ERROR = 2

_RESULT_KEYS = (
    "unexpected_count",
    "unexpected_percent",
    "missing_count",
    "missing_percent",
    "unexpected_list",
)


def _rule_label(expectation_config) -> str:
    meta = getattr(expectation_config, "meta", None) or {}
    return meta.get("rule") or expectation_config.type


def _summarize(result_detail) -> str:
    """Resumo legível das estatísticas de falha de uma expectation."""
    payload = dict(result_detail.result or {})
    parts: list[str] = []
    for key in _RESULT_KEYS:
        value = payload.get(key)
        if value is None or value == 0:
            continue
        if key in {"unexpected_percent", "missing_percent"}:
            parts.append(f"{key}={float(value):.2f}%")
        elif key == "unexpected_list":
            sample = list(value)[:5]
            parts.append(f"exemplos={sample}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def run_validation(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"[erro] Arquivo não encontrado: {csv_path}", file=sys.stderr)
        return EXIT_ERROR

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[erro] Falha ao ler o CSV: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"Contrato : {SUITE_NAME}")
    print(f"Arquivo  : {csv_path}")
    print(f"Lote     : {len(frame)} linhas x {frame.shape[1]} colunas")
    print("-" * 78)

    result = validate_dataframe(frame)
    details = sorted(result.results, key=lambda detail: detail.success)
    failures = [detail for detail in details if not detail.success]

    for detail in details:
        status = "PASS" if detail.success else "FAIL"
        label = _rule_label(detail.expectation_config)
        summary = _summarize(detail)
        print(f"[{status}] {label}" + (f"  ->  {summary}" if summary else ""))

    print("-" * 78)
    total = len(result.results)
    print(f"Regras: {total - len(failures)}/{total} aprovadas")

    if result.success:
        print("RESULTADO: PASS — ingestão LIBERADA (exit code 0)")
        return EXIT_OK

    violated = sorted(_rule_label(detail.expectation_config) for detail in failures)
    print(f"Regras violadas: {', '.join(violated)}")
    print("RESULTADO: FAIL — ingestão BLOQUEADA (exit code 1)")
    return EXIT_BLOCKED


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida um CSV contra o contrato de dados (exit 1 = ingestão bloqueada)."
    )
    parser.add_argument("csv", type=Path, help="Caminho do CSV do lote.")
    args = parser.parse_args()
    return run_validation(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
