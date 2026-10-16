"""Etapa 2: drift de features (JSON + HTML), após validação do contrato.

Duas medidas por feature:
  - KS (Kolmogorov-Smirnov), via Evidently: é o que decide se a coluna tem
    drift (p_value < limite).
  - PSI (Population Stability Index), calculado aqui: entra como leitura
    complementar. Com amostras grandes (o holdout tem ~24k linhas) o KS acusa
    diferenças pequenas como significativas; o PSI mede o *tamanho* da mudança
    e é a métrica que o mercado de crédito costuma usar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift

from credito.contrato_dados import PERFIL_INFERENCIA, PERFIL_TREINO, validar_dataframe
from credito.esquema import FEATURES
from credito.relatorio_drift import gerar_html_drift

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_REPORT = RAIZ / "artifacts" / "drift_report.json"
DEFAULT_HTML = RAIZ / "reports" / "drift_report.html"
MIN_ROWS = 100
P_VALUE = 0.05
DRIFT_SHARE = 0.5

# Faixas de leitura do PSI mais citadas (regra de bolso, não lei):
# < 0.1 estável, 0.1 a 0.25 mudança moderada, >= 0.25 mudança forte.
PSI_MODERADO = 0.1
PSI_FORTE = 0.25
PSI_BINS = 10

# Mesmo padrão do validador: 0 aprovado, 1 contrato reprovado, 2 erro.
# 3 distingue drift estatístico de lote inválido.
EXIT_STABLE = 0
EXIT_INVALID = 1
EXIT_ERROR = 2
EXIT_DRIFT = 3


def _validar_bins(bins: object) -> None:
    if (
        isinstance(bins, (bool, np.bool_))
        or not isinstance(bins, (int, np.integer))
        or bins < 2
    ):
        raise ValueError("bins deve ser um inteiro >= 2")


def _validar_serie(serie: object, nome: str) -> None:
    if getattr(serie, "ndim", 1) != 1:
        raise ValueError(f"{nome}: a entrada deve ser unidimensional")
    if isinstance(serie, pd.Series) and serie.empty:
        raise ValueError(f"{nome}: série vazia")
    if not pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie):
        raise ValueError(f"{nome}: coluna não numérica")
    if pd.api.types.is_complex_dtype(serie):
        raise ValueError(f"{nome}: coluna não numérica (complexa)")
    valores = np.asarray(serie, dtype=float)
    if valores.size == 0:
        raise ValueError(f"{nome}: série vazia")
    if not np.isfinite(valores).all():
        raise ValueError(f"{nome}: contém NaN ou infinito; use valores finitos")


def calcular_psi(
    referencia: pd.Series, atual: pd.Series, bins: int = PSI_BINS
) -> float:
    """PSI = soma((%atual - %ref) * ln(%atual / %ref)) nas faixas da referência.

    As faixas são os quantis da referência. Em colunas de contagem (atrasos,
    dependentes) vários quantis caem no mesmo valor (ex.: quase tudo é 0),
    então removo as bordas repetidas e fico com menos faixas. Exceção: se a
    referência é constante, os quantis não geram faixas úteis, então uso três
    faixas explícitas (<, == e > o valor) para o deslocamento aparecer nos
    dois sentidos.
    """
    _validar_bins(bins)
    _validar_serie(referencia, "referência")
    _validar_serie(atual, "atual")
    valores_ref = np.asarray(referencia, dtype=float)
    valores_atual = np.asarray(atual, dtype=float)
    if np.all(valores_ref == valores_ref[0]):
        constante = valores_ref[0]
        massa_ref = np.array([0.0, 1.0, 0.0])
        massa_atual = np.array(
            [
                np.count_nonzero(valores_atual < constante),
                np.count_nonzero(valores_atual == constante),
                np.count_nonzero(valores_atual > constante),
            ],
            dtype=float,
        ) / len(valores_atual)
    else:
        quantis = np.quantile(valores_ref, np.linspace(0, 1, bins + 1))
        bordas = np.unique(quantis[1:-1])
        # right=True -> faixas "até a borda": (..., b1], (b1, b2], ..., (bn, ...)
        faixa_ref = np.digitize(valores_ref, bordas, right=True)
        faixa_atual = np.digitize(valores_atual, bordas, right=True)
        total_faixas = len(bordas) + 1
        massa_ref = np.bincount(faixa_ref, minlength=total_faixas) / len(valores_ref)
        massa_atual = (
            np.bincount(faixa_atual, minlength=total_faixas) / len(valores_atual)
        )
    # Evita log(0) em faixa vazia e normaliza para as massas somarem 1.
    pct_ref = np.clip(massa_ref, 1e-6, None)
    pct_atual = np.clip(massa_atual, 1e-6, None)
    pct_ref = pct_ref / pct_ref.sum()
    pct_atual = pct_atual / pct_atual.sum()
    return float(np.sum((pct_atual - pct_ref) * np.log(pct_atual / pct_ref)))


def nivel_psi(psi: float) -> str:
    if psi >= PSI_FORTE:
        return "forte"
    if psi >= PSI_MODERADO:
        return "moderado"
    return "estavel"


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


def _gravar_html(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
    p_value: float = P_VALUE,
    drift_share: float = DRIFT_SHARE,
) -> None:
    """Gera o HTML em temporário do mesmo diretório e substitui atomicamente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".html", delete=False
        ) as output:
            temporary = Path(output.name)
        gerar_html_drift(
            reference, current, temporary, p_value=p_value, drift_share=drift_share
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
    html_path: Path | None = None,
) -> int:
    """Escreve relatório (JSON e, se pedido, HTML) e devolve o exit code.

    0=estável, 1=inválido, 2=erro, 3=drift.
    """
    if report_path.resolve() in (reference_path.resolve(), current_path.resolve()):
        raise ValueError(
            "O caminho do relatório deve ser diferente dos CSVs de entrada"
        )
    if html_path is not None and html_path.resolve() in (
        reference_path.resolve(),
        current_path.resolve(),
    ):
        raise ValueError(
            "O caminho do HTML deve ser diferente dos CSVs de entrada"
        )
    if html_path is not None and html_path.resolve() == report_path.resolve():
        raise ValueError("O caminho do HTML deve ser diferente do relatório")
    # Uma falha posterior não deve deixar um resultado de execução anterior utilizável.
    report_path.unlink(missing_ok=True)
    if html_path is not None:
        html_path.unlink(missing_ok=True)
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
            "metrica_complementar": "psi (não decide o exit code)",
            "psi_faixas": {"moderado": PSI_MODERADO, "forte": PSI_FORTE},
        },
    }

    # A referência inválida é erro de configuração, não lote inválido.
    reference_result = validar_dataframe(reference, PERFIL_TREINO)
    if not reference_result.success:
        raise ValueError("Referência não passou no contrato de dados")

    # O lote é dado de produção: pode vir sem rótulo ou já com as predições.
    current_result = validar_dataframe(current, PERFIL_INFERENCIA)
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
        psi = calcular_psi(reference[column], current[column])
        columns[column] = {
            "p_value": score,
            "drift": score < p_value,
            "psi": round(psi, 4),
            "psi_nivel": nivel_psi(psi),
        }
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
    if html_path is not None:
        _gravar_html(reference, current, html_path, p_value, drift_share)
    _gravar(report_path, payload)
    return EXIT_DRIFT if detected else EXIT_STABLE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida lote e detecta drift de features."
    )
    parser.add_argument("lote", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--html",
        type=Path,
        default=DEFAULT_HTML,
        help="Destino do relatório HTML do Evidently (não usado em erro).",
    )
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
            args.html,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    labels = {
        EXIT_STABLE: "sem_drift",
        EXIT_INVALID: "lote_invalido",
        EXIT_ERROR: "erro",
        EXIT_DRIFT: "drift_detectado",
    }
    if code in (EXIT_STABLE, EXIT_DRIFT):
        _imprimir_colunas(args.report)
    message = f"RESULTADO: {labels[code]} (exit code {code}); relatório: {args.report}"
    if code in (EXIT_STABLE, EXIT_DRIFT):
        message += f"; HTML: {args.html}"
    print(message)
    return code


def _imprimir_colunas(report_path: Path) -> None:
    """Tabelinha no console com KS e PSI de cada feature (lida do JSON)."""
    colunas = json.loads(report_path.read_text(encoding="utf-8"))["colunas"]
    print(f"{'feature':<40} {'p_value KS':>11} {'drift':>6} {'PSI':>8}  nível PSI")
    for nome, info in colunas.items():
        drift = "sim" if info["drift"] else "não"
        print(
            f"{nome:<40} {info['p_value']:>11.4g} {drift:>6} "
            f"{info['psi']:>8.4f}  {info['psi_nivel']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
