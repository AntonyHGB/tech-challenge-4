"""Etapa 3: pipeline monitorado — fluxo completo sob MLflow local (SQLite).

Usa MLflow real (SQLite em tmp_path) e GX real; nada de backend mockado.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import construir_lote
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pandas.testing import assert_frame_equal

from credito.modelo import dividir_referencia
from credito.pipeline_monitorado import executar_pipeline, main

ARTEFATOS_ESPERADOS = {
    "resumo.json",
    "baseline_metrics.json",
    "drift_report.json",
    "drift_report.html",
}


@pytest.fixture
def referencia_csv(tmp_path: Path) -> Path:
    """Referência com 600 linhas -> holdout de 120 (> min_rows 100 do drift)."""
    caminho = tmp_path / "entrada" / "reference.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    construir_lote(600).to_csv(caminho, index=False)
    return caminho


def _run_unico(output: Path) -> tuple:
    caminho_banco = (output / "mlflow.db").as_posix()
    cliente = MlflowClient(tracking_uri=f"sqlite:///{caminho_banco}")
    experimento = cliente.get_experiment_by_name("credito-pipeline")
    assert experimento is not None
    runs = cliente.search_runs([experimento.experiment_id])
    assert len(runs) == 1
    return runs[0], cliente


def test_estavel_fluxo_completo(tmp_path, referencia_csv):
    output = tmp_path / "monitoramento"
    codigo = executar_pipeline(referencia_csv, output, cenario="estavel", estagio=3)
    assert codigo == 0
    assert (output / "mlflow.db").exists()

    run, cliente = _run_unico(output)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resultado"] == "sem_drift"
    assert run.data.tags["codigo_saida"] == "0"
    assert run.data.params["cenario"] == "estavel"
    assert run.data.params["estagio"] == "3"

    run_dir = output / "runs" / run.info.run_id
    resumo = json.loads((run_dir / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["run_id"] == run.info.run_id
    assert resumo["resultado"] == "sem_drift" and resumo["codigo_saida"] == 0
    assert resumo["volumes"] == {"n_total": 600, "n_treino": 480, "n_holdout": 120}

    # Mesmo split do treino reusado: produção = holdout, referência = treino.
    frame = pd.read_csv(referencia_csv)
    treino_esperado, holdout_esperado = dividir_referencia(frame)
    treino_salvo = pd.read_csv(run_dir / "referencia_treino.csv")
    lote_salvo = pd.read_csv(run_dir / "lote_estavel.csv")
    assert_frame_equal(
        treino_salvo,
        treino_esperado.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )
    assert_frame_equal(
        lote_salvo,
        holdout_esperado.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )
    chaves_treino = set(map(tuple, treino_salvo.to_numpy().tolist()))
    chaves_lote = set(map(tuple, lote_salvo.to_numpy().tolist()))
    assert chaves_treino.isdisjoint(chaves_lote)

    # CSVs e modelo nunca viram artefato MLflow; só os arquivos-resultado.
    artefatos = {info.path for info in cliente.list_artifacts(run.info.run_id)}
    assert artefatos >= ARTEFATOS_ESPERADOS
    assert "model.joblib" not in artefatos
    assert not any(caminho.endswith(".csv") for caminho in artefatos)

    metricas = run.data.metrics
    assert any("modelo/baseline" in chave and "roc_auc" in chave for chave in metricas)
    assert any("modelo/producao" in chave and "roc_auc" in chave for chave in metricas)
    assert any("volume" in chave and "entrada_linhas" in chave for chave in metricas)
    assert any("volume" in chave and "holdout_linhas" in chave for chave in metricas)
    assert any("drift" in chave and "share" in chave for chave in metricas)
    assert any("drift" in chave and "psi" in chave for chave in metricas)
    assert any("etapa" in chave and "duracao" in chave for chave in metricas)
    assert not any("alertas" in chave for chave in metricas)  # estável: sem alerta

    relatorio = json.loads((run_dir / "drift_report.json").read_text(encoding="utf-8"))
    assert relatorio["status"] == "sem_drift"
    assert (run_dir / "predicoes.csv").exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "logs.jsonl").exists()


def test_drift_detectado_t3(tmp_path, referencia_csv):
    output = tmp_path / "monitoramento"
    codigo = executar_pipeline(referencia_csv, output, cenario="drift", estagio=3)
    assert codigo == 3

    run, _ = _run_unico(output)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resultado"] == "drift_detectado"
    assert run.data.tags["codigo_saida"] == "3"
    assert any(
        chave.startswith("alertas") and "drift" in chave for chave in run.data.metrics
    )

    run_dir = output / "runs" / run.info.run_id
    relatorio = json.loads((run_dir / "drift_report.json").read_text(encoding="utf-8"))
    assert relatorio["status"] == "drift_detectado"
    assert relatorio["agregado"]["share"] >= 0.5
    resumo = json.loads((run_dir / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["resultado"] == "drift_detectado" and resumo["codigo_saida"] == 3

    # O lote do drift NÃO é o holdout intocado (t3 desloca renda e idade).
    lote = pd.read_csv(run_dir / "lote_drift.csv")
    frame = pd.read_csv(referencia_csv)
    _, holdout = dividir_referencia(frame)
    assert not lote["MonthlyIncome"].equals(
        holdout.reset_index(drop=True)["MonthlyIncome"]
    )
    assert (run_dir / "predicoes.csv").exists()


def test_invalido_bloqueia_antes_da_predicao(tmp_path, referencia_csv):
    output = tmp_path / "monitoramento"
    codigo = executar_pipeline(referencia_csv, output, cenario="invalido", estagio=3)
    assert codigo == 1

    run, cliente = _run_unico(output)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resultado"] == "bloqueado"
    assert run.data.tags["codigo_saida"] == "1"
    assert any(
        chave.startswith("alertas") and "qualidade" in chave
        for chave in run.data.metrics
    )

    run_dir = output / "runs" / run.info.run_id
    qualidade = json.loads((run_dir / "qualidade.json").read_text(encoding="utf-8"))
    assert set(qualidade) == {"regras_violadas", "contagem"}
    assert qualidade["contagem"] == len(qualidade["regras_violadas"]) > 0
    assert all(isinstance(regra, str) for regra in qualidade["regras_violadas"])
    assert not (run_dir / "predicoes.csv").exists()
    assert not (run_dir / "drift_report.json").exists()
    assert (run_dir / "baseline_metrics.json").exists()  # treino veio antes
    resumo = json.loads((run_dir / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["resultado"] == "bloqueado" and resumo["codigo_saida"] == 1
    assert resumo["volumes"]["n_holdout"] == 120
    artefatos = {info.path for info in cliente.list_artifacts(run.info.run_id)}
    assert "drift_report.json" not in artefatos
    assert "resumo.json" in artefatos
    assert "qualidade.json" in artefatos  # agregado, sem PII e sem run_dir
    destino = tmp_path / "artefatos_baixados"
    cliente.download_artifacts(run.info.run_id, "qualidade.json", str(destino))
    baixado = json.loads((destino / "qualidade.json").read_text(encoding="utf-8"))
    assert baixado == qualidade


def test_entrada_ausente_erro_persistido_sem_traceback(tmp_path, capsys):
    output = tmp_path / "monitoramento"
    codigo = executar_pipeline(tmp_path / "ausente.csv", output, cenario="estavel")
    assert codigo == 2

    run, _ = _run_unico(output)
    assert run.info.status == "FAILED"
    assert run.data.tags["resultado"] == "erro"
    assert run.data.tags["codigo_saida"] == "2"

    run_dir = output / "runs" / run.info.run_id
    assert not (run_dir / "resumo.json").exists()  # resumo só em saída terminal
    logs = (run_dir / "logs.jsonl").read_text(encoding="utf-8")
    assert "execucao_erro" in logs and "FileNotFoundError" in logs
    logs_eventos = [json.loads(linha) for linha in logs.splitlines()]
    assert all("ausente" not in json.dumps(evento) for evento in logs_eventos)

    _, cliente_erro = _run_unico(output)
    artefatos = {info.path for info in cliente_erro.list_artifacts(run.info.run_id)}
    assert "logs.jsonl" in artefatos

    saida_erro = capsys.readouterr().err
    assert "FileNotFoundError" in saida_erro
    assert "Traceback" not in saida_erro


def test_cli_monitorar_pipeline(tmp_path, referencia_csv, monkeypatch):
    output = tmp_path / "monitoramento"
    monkeypatch.setattr(
        "sys.argv",
        [
            "monitorar_pipeline.py",
            "--input",
            str(referencia_csv),
            "--output",
            str(output),
            "--cenario",
            "estavel",
            "--estagio",
            "1",
        ],
    )
    assert main() == 0
    run, _ = _run_unico(output)
    assert run.data.tags["codigo_saida"] == "0"
    assert run.data.params["estagio"] == "1"


def test_script_monitorar_pipeline_help(tmp_path):
    """O script real existe: --help via subprocesso sai 0 com --cenario/--output."""
    raiz = Path(__file__).resolve().parents[1]
    resultado = subprocess.run(
        [sys.executable, str(raiz / "scripts" / "monitorar_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "--cenario" in resultado.stdout
    assert "--output" in resultado.stdout


def test_mlflow_exception_vira_erro_2_sem_traceback(
    tmp_path, referencia_csv, monkeypatch, capsys
):
    """MlflowException após __enter__: exit 2, run FAILED, stderr sem traceback."""
    def log_param_explodindo(*args, **kwargs):
        raise MlflowException("falha simulada")

    monkeypatch.setattr(MlflowClient, "log_param", log_param_explodindo)
    output = tmp_path / "monitoramento"
    codigo = executar_pipeline(referencia_csv, output, cenario="estavel", estagio=1)
    assert codigo == 2

    run, _ = _run_unico(output)
    assert run.info.status == "FAILED"
    assert run.data.tags["resultado"] == "erro"
    assert run.data.tags["codigo_saida"] == "2"

    capturado = capsys.readouterr()
    linha_id = next(
        linha
        for linha in capturado.out.splitlines()
        if linha.startswith("[monitoramento] run_id=")
    )
    assert linha_id == f"[monitoramento] run_id={run.info.run_id}"

    assert "MlflowException" in capturado.err
    assert "Traceback" not in capturado.err
