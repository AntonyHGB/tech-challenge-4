"""Etapa 3 — Pipeline monitorado: baseline + produção simulada + drift no MLflow.

Orquestra, em um único run do MLflow local (SQLite), as etapas existentes do
projeto — preparação, contrato de dados, treino, simulação de produção,
predição e comparação de drift — reutilizando os módulos da Etapas 1 e 2 e a
classe ``Monitoramento`` (observabilidade). Nenhum dado de cliente e nenhuma
mensagem bruta de exceção vai para os logs operacionais.

Cenários do lote de produção (sempre derivado do holdout, nunca visto no treino):
  - ``estavel`` : o holdout intocado, agindo como um "t0" fingido;
  - ``drift``   : o último estágio de ``simular_producao`` (passagem do tempo);
  - ``invalido``: o holdout com uma idade fora da faixa do contrato (bloqueio).

Exit codes: 0 estável, 1 bloqueado (qualidade), 2 erro, 3 drift detectado.

Uso:
    python scripts/monitorar_pipeline.py
    python scripts/monitorar_pipeline.py --cenario estavel --estagio 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.exceptions import MlflowException

from credito.contrato_dados import PERFIL_INFERENCIA, PERFIL_TREINO, validar_dataframe
from credito.drift import comparar
from credito.modelo import dividir_referencia, treinar
from credito.observabilidade import Monitoramento
from credito.predicao import predizer
from credito.preparacao import preparar_referencia
from credito.simulacao_producao import simular_producao

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_OUTPUT = RAIZ / "artifacts" / "monitoramento"

SEED = 42
ESTAGIO_PADRAO = 3
P_VALUE_LIMITE = 0.05
DRIFT_SHARE = 0.5

CENARIOS = ("estavel", "drift", "invalido")
STATUS_DRIFT = ("sem_drift", "drift_detectado", "lote_invalido", "amostra_insuficiente")

CODIGO_ESTAVEL = 0
CODIGO_BLOQUEADO = 1
CODIGO_ERRO = 2
CODIGO_DRIFT = 3

ARTEFATOS_RESULTADO = (
    "baseline_metrics.json",
    "drift_report.json",
    "drift_report.html",
    "qualidade.json",
)


def _numericas(dados: dict[str, Any]) -> dict[str, float]:
    """Só escalares numéricos finitos: listas, textos e NaN ficam de fora."""
    return {
        chave: float(valor)
        for chave, valor in dados.items()
        if isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(float(valor))
    }


def _regras_violadas(resultado: Any) -> list[str]:
    """Rótulos das regras violadas (sem valores/linhas — não vaza dado de cliente)."""
    nomes: list[str] = []
    for detalhe in resultado.results:
        if detalhe.success:
            continue
        config = detalhe.expectation_config
        meta = getattr(config, "meta", None) or {}
        nomes.append(str(meta.get("rule") or config.type))
    return sorted(nomes)


def _salvar_qualidade(run_dir: Path, violadas: list[str]) -> None:
    """JSON operacional do bloqueio: nomes de regras e contagem, nada além."""
    payload = {"regras_violadas": violadas, "contagem": len(violadas)}
    (run_dir / "qualidade.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _montar_lote(cenario: str, estagio: int, holdout: pd.DataFrame) -> pd.DataFrame:
    """Monta o lote de produção do cenário a partir do holdout intocado."""
    if cenario == "estavel":
        return holdout.copy()
    if cenario == "invalido":
        lote = holdout.copy()
        lote.loc[lote.index[0], "age"] = 16  # abaixo do mínimo do contrato
        return lote
    if cenario == "drift":
        return simular_producao(holdout, seed=SEED, estagios=estagio)[-1][1]
    raise ValueError(f"cenario desconhecido: {cenario!r} (use um de {CENARIOS})")


def _registrar_metricas_drift(mon: Monitoramento, relatorio: dict[str, Any]) -> None:
    """Share agregado e p_value/PSI por feature, com keys estáveis."""
    agregado = relatorio.get("agregado") or {}
    metricas: dict[str, Any] = {
        chave: agregado[chave]
        for chave in ("share", "colunas_com_drift", "total_colunas")
        if chave in agregado
    }
    for coluna, info in (relatorio.get("colunas") or {}).items():
        metricas[f"{coluna}/p_value"] = info.get("p_value")
        metricas[f"{coluna}/psi"] = info.get("psi")
    mon.registrar_metricas(_numericas(metricas), prefixo="drift/")


def _finalizar(
    mon: Monitoramento,
    cenario: str,
    estagio: int,
    status: str,
    codigo: int,
    volumes: dict[str, int | None],
) -> None:
    """Grava o resumo do run e envia os artefatos-resultado ao MLflow."""
    resumo = {
        "run_id": mon.run_id,
        "cenario": cenario,
        "estagio": estagio,
        "resultado": status,
        "codigo_saida": codigo,
        "volumes": volumes,
    }
    caminho_resumo = mon.diretorio_execucao / "resumo.json"
    caminho_resumo.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mon.registrar_artefato(caminho_resumo)
    for nome in ARTEFATOS_RESULTADO:
        alvo = mon.diretorio_execucao / nome
        if alvo.is_file():
            mon.registrar_artefato(alvo)


def _fluxo(mon: Monitoramento, input_csv: Path, cenario: str, estagio: int) -> int:
    """Pipeline completo sob um run; devolve o exit code da execução."""
    run_dir = mon.diretorio_execucao
    volumes: dict[str, int | None] = {
        "n_total": None,
        "n_treino": None,
        "n_holdout": None,
    }

    with mon.etapa("ingestao"):
        if not input_csv.exists():
            raise FileNotFoundError(
                f"{input_csv} não encontrado; informe um --input válido"
            )
        frame = pd.read_csv(input_csv)
        volumes["n_total"] = len(frame)
        mon.registrar_metricas({"entrada_linhas": len(frame)}, prefixo="volume/")

    with mon.etapa("preparacao"):
        preparar_referencia(input_csv, run_dir / "reference.csv")
        frame = pd.read_csv(run_dir / "reference.csv")
        mon.registrar_metricas({"preparadas_linhas": len(frame)}, prefixo="volume/")

    with mon.etapa("validacao_referencia"):
        resultado_ref = validar_dataframe(frame, PERFIL_TREINO)
        if not resultado_ref.success:
            violadas = ", ".join(_regras_violadas(resultado_ref))
            raise ValueError(f"Referência fora do contrato (perfil treino): {violadas}")

    with mon.etapa("treino"):
        metricas_baseline = treinar(
            run_dir / "reference.csv",
            run_dir / "model.joblib",
            run_dir / "baseline_metrics.json",
        )
        mon.registrar_metricas(
            _numericas(metricas_baseline), prefixo="modelo/baseline/"
        )

    with mon.etapa("simulacao"):
        # Mesmo split determinístico do treino: produção = holdout puro.
        treino, holdout = dividir_referencia(frame)
        treino = treino.reset_index(drop=True)
        holdout = holdout.reset_index(drop=True)
        treino.to_csv(run_dir / "referencia_treino.csv", index=False)
        nome_lote = f"lote_{cenario}.csv"
        _montar_lote(cenario, estagio, holdout).to_csv(run_dir / nome_lote, index=False)
        volumes["n_treino"] = len(treino)
        volumes["n_holdout"] = len(holdout)
        mon.registrar_metricas(
            {"treino_linhas": len(treino), "holdout_linhas": len(holdout)},
            prefixo="volume/",
        )

    with mon.etapa("validacao_producao"):
        resultado_lote = validar_dataframe(
            _ler_csv(run_dir / nome_lote), PERFIL_INFERENCIA
        )
        if not resultado_lote.success:
            violadas = _regras_violadas(resultado_lote)
            _salvar_qualidade(run_dir, violadas)
            mon.registrar_metricas(
                {"regras_violadas": len(violadas)}, prefixo="qualidade/"
            )
            mon.alerta(
                "qualidade", "lote reprovado no contrato de dados (perfil inferencia)"
            )
            mon.resultado("bloqueado", CODIGO_BLOQUEADO)
            _finalizar(mon, cenario, estagio, "bloqueado", CODIGO_BLOQUEADO, volumes)
            return CODIGO_BLOQUEADO

    with mon.etapa("predicao"):
        metricas_producao = predizer(
            run_dir / nome_lote, run_dir / "model.joblib", run_dir / "predicoes.csv"
        )
        if metricas_producao is not None:
            mon.registrar_metricas(
                _numericas(metricas_producao), prefixo="modelo/producao/"
            )

    with mon.etapa("drift"):
        comparar(
            run_dir / "referencia_treino.csv",
            run_dir / nome_lote,
            run_dir / "drift_report.json",
            html_path=run_dir / "drift_report.html",
        )
        relatorio = json.loads(
            (run_dir / "drift_report.json").read_text(encoding="utf-8")
        )
        _registrar_metricas_drift(mon, relatorio)
        return _desfecho_drift(mon, relatorio, cenario, estagio, volumes)


def _desfecho_drift(
    mon: Monitoramento,
    relatorio: dict[str, Any],
    cenario: str,
    estagio: int,
    volumes: dict[str, int | None],
) -> int:
    """Mapeia o status do relatório de drift para resultado/exit code."""
    status = str(relatorio.get("status", ""))
    if status == "drift_detectado":
        agregado = relatorio.get("agregado") or {}
        mon.alerta(
            "drift",
            f"drift detectado: share {float(agregado.get('share', 0.0)):.2f} >= limite",
        )
        mon.resultado("drift_detectado", CODIGO_DRIFT)
        _finalizar(mon, cenario, estagio, "drift_detectado", CODIGO_DRIFT, volumes)
        return CODIGO_DRIFT
    if status == "lote_invalido":
        mon.alerta("qualidade", "lote reprovado no contrato durante a comparação")
        mon.resultado("bloqueado", CODIGO_BLOQUEADO)
        _finalizar(mon, cenario, estagio, "bloqueado", CODIGO_BLOQUEADO, volumes)
        return CODIGO_BLOQUEADO
    if status == "amostra_insuficiente":
        mon.resultado("erro", CODIGO_ERRO)
        _finalizar(mon, cenario, estagio, "erro", CODIGO_ERRO, volumes)
        return CODIGO_ERRO
    if status == "sem_drift":
        mon.resultado("sem_drift", CODIGO_ESTAVEL)
        _finalizar(mon, cenario, estagio, "sem_drift", CODIGO_ESTAVEL, volumes)
        return CODIGO_ESTAVEL
    raise ValueError(
        f"status de drift inesperado: {status!r} (esperados: {STATUS_DRIFT})"
    )


def _ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


def _validar_argumentos(cenario: str, estagio: int) -> None:
    if cenario not in CENARIOS:
        raise ValueError(f"cenario desconhecido: {cenario!r} (use um de {CENARIOS})")
    if not 1 <= estagio <= 3:
        raise ValueError(f"estagio deve estar entre 1 e 3 (recebido: {estagio})")


def _executar(
    input_csv: Path, output_dir: Path, cenario: str, estagio: int
) -> int:
    with Monitoramento(output_dir, f"pipeline-{cenario}-t{estagio}") as mon:
        print(f"[monitoramento] run_id={mon.run_id}")
        cliente = mon.client
        if cliente is None:  # o __enter__ sempre define o cliente
            raise RuntimeError("run iniciado sem cliente do MLflow")
        cliente.log_param(mon.run_id, "cenario", cenario)
        cliente.log_param(mon.run_id, "estagio", estagio)
        cliente.log_param(mon.run_id, "seed", SEED)
        cliente.log_param(mon.run_id, "p_value_limite", P_VALUE_LIMITE)
        cliente.log_param(mon.run_id, "drift_share", DRIFT_SHARE)
        codigo = _fluxo(mon, input_csv, cenario, estagio)
    return codigo


def executar_pipeline(
    input_csv: Path,
    output_dir: Path,
    cenario: str = "drift",
    estagio: int = ESTAGIO_PADRAO,
) -> int:
    """Roda o pipeline monitorado e devolve o exit code (0/1/2/3).

    Erros esperados (entrada ausente, contrato, CSV ilegível) viram exit 2 com
    mensagem clara no stderr, sem traceback; o run é marcado FAILED pelo
    ``Monitoramento`` e os logs ficam persistidos.
    """
    try:
        _validar_argumentos(cenario, estagio)
        return _executar(input_csv, output_dir, cenario, estagio)
    except (
        OSError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
        MlflowException,
    ) as exc:
        print(f"[erro] {type(exc).__name__}: {exc}", file=sys.stderr)
        return CODIGO_ERRO


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roda o pipeline monitorado com MLflow local (exit 3 = drift)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV de referência de entrada.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Diretório do monitoramento (mlflow.db e runs/).",
    )
    parser.add_argument(
        "--cenario",
        choices=CENARIOS,
        default="drift",
        help="Cenário do lote de produção.",
    )
    parser.add_argument(
        "--estagio",
        type=int,
        default=ESTAGIO_PADRAO,
        help="Estágio da simulação de produção (1 a 3).",
    )
    args = parser.parse_args()
    return executar_pipeline(
        args.input, args.output, cenario=args.cenario, estagio=args.estagio
    )


if __name__ == "__main__":
    raise SystemExit(main())
