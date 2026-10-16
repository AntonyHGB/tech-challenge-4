"""Etapa 3: observabilidade local — MLflow (SQLite) + logs JSONL e métricas da máquina.

Fluxo simples e documentado:
    with Monitoramento(diretorio, "nome-execucao") as mon:
        with mon.etapa("preparacao"):
            mon.registrar_metricas({"linhas": n})
            mon.registrar_artefato(mon.diretorio_execucao / "arquivo.csv")
            mon.alerta("drift", "desvio detectado")
        mon.resultado("drift", 3)  # opcional; sem exceção o padrão é ("ok", 0)

Exceção inesperada dentro do ``with`` registra resultado ("erro", 2), status
MLflow FAILED e não é engolida. Falha ao persistir logs/artefato no desfecho
segue o mesmo rumo: resultado ("erro", 2), FAILED e exceção preservada. Banco
SQLite em ``diretorio/mlflow.db`` e
artefatos em ``diretorio/mlartifacts``; eventos operacionais em
``diretorio/runs/<run_id>/logs.jsonl`` (sem dados de clientes, segredos,
mensagens brutas de exceção ou paths absolutos).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

_ARQUIVO_LOGS = "logs.jsonl"


class Monitoramento:
    """Execução observada: MLflow local (SQLite) + JSONL central de eventos."""

    def __init__(
        self, diretorio: Path, nome_execucao: str, experimento: str = "credito-pipeline"
    ) -> None:
        self.diretorio = Path(diretorio).resolve()
        self.nome_execucao = nome_execucao
        self.experimento = experimento
        self.run_id = ""
        self.diretorio_execucao = self.diretorio / "runs"
        self.client: MlflowClient | None = None
        self._pilha_etapas: list[str] = []
        self._alertas: dict[str, int] = {}
        self._resultado_definido = False
        self._excecao_alertada: BaseException | None = None

    def __enter__(self) -> Monitoramento:
        self.diretorio.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(self.diretorio / 'mlflow.db').as_posix()}"
        cliente = MlflowClient(tracking_uri=tracking_uri)
        experimento = cliente.get_experiment_by_name(self.experimento)
        if experimento is None:
            artifact_location = (self.diretorio / "mlartifacts").as_uri()
            experiment_id = cliente.create_experiment(
                self.experimento, artifact_location=artifact_location
            )
        else:
            experiment_id = experimento.experiment_id
        run = cliente.create_run(experiment_id, run_name=self.nome_execucao)
        self.client = cliente
        self.run_id = run.info.run_id
        self.diretorio_execucao = self.diretorio / "runs" / self.run_id
        self.diretorio_execucao.mkdir(parents=True)
        self._escrever_evento("execucao_iniciada", status="ok")
        return self

    def __exit__(self, tipo_exc: object, exc: BaseException | None, tb: object) -> bool:
        if exc is None:
            if not self._resultado_definido:
                self.resultado("ok", 0)
            try:
                self._registrar_logs_artefato()
            except (OSError, MlflowException) as falha:
                # persistência falhou: sem falso sucesso (run RUNNING/ok/0)
                self._finalizar_erro_persistencia(falha)
                raise
            self._cliente().set_terminated(self.run_id, "FINISHED")
            return False  # nada a engolir
        try:
            if exc is not self._excecao_alertada:
                self.alerta("falha", type(exc).__name__)
            self._escrever_evento(
                "execucao_erro", status="falha", erro=type(exc).__name__
            )
            self.resultado("erro", 2)
            self._registrar_logs_artefato()
        except Exception:  # preserva a exceção original; sem falso sucesso
            pass
        with suppress(Exception):
            self._cliente().set_terminated(self.run_id, "FAILED")
        return False  # não engole a exceção

    @contextmanager
    def etapa(self, nome: str) -> Iterator[Monitoramento]:
        """Registra início/fim/duração da etapa com snapshots de máquina."""
        self._pilha_etapas.append(nome)
        inicio = time.perf_counter()
        snapshot = self._log_metricas(
            self._snapshot_maquina(), prefixo=f"etapa.{nome}.inicio."
        )
        self._escrever_evento("etapa_iniciada", status="ok", metricas=snapshot)
        try:
            yield self
        except BaseException as exc:  # registra falha e preserva a exceção
            self._escrever_evento(
                "etapa_falhou", status="falha", erro=type(exc).__name__
            )
            self.alerta("falha", type(exc).__name__)
            self._excecao_alertada = exc
            raise
        else:
            fim = self._log_metricas(
                self._snapshot_maquina(), prefixo=f"etapa.{nome}.fim."
            )
            fim.update(
                self._log_metricas(
                    {"duracao_segundos": time.perf_counter() - inicio},
                    prefixo=f"etapa.{nome}.",
                )
            )
            self._escrever_evento("etapa_finalizada", status="ok", metricas=fim)
        finally:
            self._pilha_etapas.pop()

    def registrar_metricas(
        self, dados: dict[str, Any], prefixo: str = ""
    ) -> dict[str, float]:
        """Métricas finitas no MLflow e no evento JSONL; None é omitido."""
        registradas = self._log_metricas(dados, prefixo)
        self._escrever_evento("metricas", status="ok", metricas=registradas)
        return registradas

    def registrar_artefato(self, caminho: Path) -> None:
        """Copia um arquivo produzido sob ``diretorio_execucao`` p/ o MLflow."""
        cliente = self._cliente()
        arquivo = Path(caminho).resolve()
        base = self.diretorio_execucao.resolve()
        if base not in arquivo.parents or not arquivo.is_file():
            raise ValueError(
                "artefato deve ser um arquivo dentro do diretorio_execucao"
            )
        cliente.log_artifact(self.run_id, str(arquivo))
        self._escrever_evento(
            "artefato", status="ok", artefato=arquivo.relative_to(base).as_posix()
        )

    def alerta(self, tipo: str, mensagem: str) -> None:
        """Evento de alerta + contagem por tipo (métrica ``alertas.<tipo>``)."""
        self._alertas[tipo] = self._alertas.get(tipo, 0) + 1
        self._cliente().log_metric(
            self.run_id, f"alertas.{tipo}", float(self._alertas[tipo])
        )
        self._escrever_evento("alerta", status="alerta", tipo=tipo, mensagem=mensagem)

    def resultado(self, status: str, codigo: int) -> None:
        """Desfecho da execução: tags ``resultado`` e ``codigo_saida`` no run."""
        cliente = self._cliente()
        cliente.set_tag(self.run_id, "resultado", status)
        cliente.set_tag(self.run_id, "codigo_saida", str(int(codigo)))
        self._resultado_definido = True
        self._escrever_evento(
            "resultado", status=status, codigo=codigo, alertas=dict(self._alertas)
        )

    def _cliente(self) -> MlflowClient:
        if self.client is None:
            raise RuntimeError("use Monitoramento como context manager (with ...)")
        return self.client

    def _snapshot_maquina(self) -> dict[str, float]:
        """CPU/RAM/disco do host (percent) e RSS do processo (bytes)."""
        disco = psutil.disk_usage(self.diretorio.anchor or "/")
        return {
            "sistema_cpu_percent": psutil.cpu_percent(interval=0.05),
            "sistema_ram_percent": psutil.virtual_memory().percent,
            "sistema_disco_percent": disco.percent,
            "sistema_rss_bytes": float(psutil.Process().memory_info().rss),
        }

    def _log_metricas(
        self, dados: dict[str, Any], prefixo: str = ""
    ) -> dict[str, float]:
        cliente = self._cliente()
        # valida TUDO antes de logar: sem batch parcial NaN/inf no MLflow
        valores: list[tuple[str, float]] = []
        for chave, valor in dados.items():
            if valor is None:
                continue
            numero = float(valor)
            if not math.isfinite(numero):
                raise ValueError(
                    f"métrica '{prefixo}{chave}' não é finita ({numero}); "
                    "NaN/±inf não são válidos em métrica nem em JSON estrito"
                )
            valores.append((f"{prefixo}{chave}", numero))
        registradas: dict[str, float] = {}
        for nome, numero in valores:
            cliente.log_metric(self.run_id, nome, numero)
            registradas[nome] = numero
        return registradas

    def _registrar_logs_artefato(self) -> None:
        self.registrar_artefato(self.diretorio_execucao / _ARQUIVO_LOGS)

    def _finalizar_erro_persistencia(self, falha: BaseException) -> None:
        """Persistência falhou no desfecho: erro real, best-effort."""
        with suppress(Exception):
            self._escrever_evento(
                "execucao_erro", status="falha", erro=type(falha).__name__
            )
        with suppress(Exception):
            self.resultado("erro", 2)
        with suppress(Exception):
            self._cliente().set_terminated(self.run_id, "FAILED")

    def _escrever_evento(self, evento: str, *, status: str, **campos: Any) -> None:
        registro: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "run_id": self.run_id,
            "etapa": self._pilha_etapas[-1] if self._pilha_etapas else None,
            "evento": evento,
            "status": status,
        }
        registro.update(campos)
        caminho_logs = self.diretorio_execucao / _ARQUIVO_LOGS
        with open(caminho_logs, "a", encoding="utf-8") as arquivo:
            arquivo.write(
                json.dumps(registro, ensure_ascii=False, allow_nan=False) + "\n"
            )
