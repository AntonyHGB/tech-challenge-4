"""Etapa 3: observabilidade local MLflow/SQLite, logs JSONL e métricas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credito.observabilidade import Monitoramento


def _ler_logs(diretorio_execucao) -> list[dict]:
    conteudo = (diretorio_execucao / "logs.jsonl").read_text(encoding="utf-8")
    return [json.loads(linha) for linha in conteudo.splitlines()]


def test_registra_etapa_metricas_jsonl_e_run_finalizado(tmp_path):
    with Monitoramento(tmp_path, "exec-baseline") as mon, mon.etapa("preparacao"):
        mon.registrar_metricas({"linhas": 100, "vazio": None})

    run = mon.client.get_run(mon.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resultado"] == "ok"
    assert run.data.tags["codigo_saida"] == "0"

    metricas = run.data.metrics
    assert metricas["linhas"] == 100.0
    assert "vazio" not in metricas  # None é omitido, nunca 0 fake
    assert 0.0 <= metricas["etapa.preparacao.inicio.sistema_cpu_percent"] <= 100.0
    assert metricas["etapa.preparacao.fim.sistema_rss_bytes"] > 0
    assert metricas["etapa.preparacao.duracao_segundos"] >= 0.0

    assert mon.diretorio_execucao == tmp_path / "runs" / mon.run_id
    assert (tmp_path / "mlflow.db").exists()

    eventos = _ler_logs(mon.diretorio_execucao)
    nomes = [evento["evento"] for evento in eventos]
    assert nomes[0] == "execucao_iniciada"
    assert "etapa_iniciada" in nomes and "etapa_finalizada" in nomes
    for evento in eventos:
        assert evento["run_id"] == mon.run_id
        assert evento["timestamp_utc"]
        assert "etapa" in evento and "status" in evento
    texto = json.dumps(eventos)
    assert str(tmp_path) not in texto  # sem paths absolutos nos eventos
    artefatos = {info.path for info in mon.client.list_artifacts(mon.run_id)}
    assert "logs.jsonl" in artefatos  # readback dos logs como artefato


def test_excecao_inesperada_marca_failed_preserva_e_alerta_unico(tmp_path):
    with (
        pytest.raises(ValueError),
        Monitoramento(tmp_path, "exec-erro") as mon,
        mon.etapa("treino"),
    ):
        raise ValueError("boom")

    run = mon.client.get_run(mon.run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags["resultado"] == "erro"
    assert run.data.tags["codigo_saida"] == "2"
    assert run.data.metrics["alertas.falha"] == 1.0  # sem alerta duplo p/ mesma exceção

    eventos = _ler_logs(mon.diretorio_execucao)
    assert [evento["evento"] for evento in eventos].count("etapa_falhou") == 1
    falha = next(evento for evento in eventos if evento["evento"] == "etapa_falhou")
    assert falha["status"] == "falha" and falha["erro"] == "ValueError"
    erros = [evento for evento in eventos if evento["evento"] == "execucao_erro"]
    assert len(erros) == 1 and erros[0]["erro"] == "ValueError"
    alertas = [evento for evento in eventos if evento["evento"] == "alerta"]
    assert len(alertas) == 1 and alertas[0]["tipo"] == "falha"
    assert "boom" not in json.dumps(eventos)  # sem mensagem bruta/traceback


def test_resultado_custom_drift_finaliza_sem_crash(tmp_path):
    with Monitoramento(tmp_path, "exec-drift") as mon:
        mon.resultado("drift", 3)

    run = mon.client.get_run(mon.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resultado"] == "drift"
    assert run.data.tags["codigo_saida"] == "3"
    eventos = _ler_logs(mon.diretorio_execucao)
    resultados = [evento for evento in eventos if evento["evento"] == "resultado"]
    assert len(resultados) == 1 and resultados[0]["status"] == "drift"


def test_runs_disjuntos_persistem_historico(tmp_path):
    with Monitoramento(tmp_path, "exec-1") as mon1:
        mon1.registrar_metricas({"linhas": 10})
    with Monitoramento(tmp_path, "exec-2") as mon2:
        mon2.registrar_metricas({"linhas": 20})

    experimento = mon2.client.get_experiment_by_name("credito-pipeline")
    assert experimento is not None
    assert "mlartifacts" in experimento.artifact_location  # artifact_location explícito
    runs = mon2.client.search_runs([experimento.experiment_id])
    assert {run.data.tags["mlflow.runName"] for run in runs} >= {"exec-1", "exec-2"}
    historia = mon1.client.get_metric_history(mon1.run_id, "linhas")
    assert [m.value for m in historia] == [10.0]
    assert mon2.client.get_run(mon1.run_id).data.metrics["linhas"] == 10.0


def test_artefato_aceita_somente_arquivo_sob_diretorio_execucao(tmp_path):
    with Monitoramento(tmp_path, "exec-artefatos") as mon:
        interno = mon.diretorio_execucao / "relatorio.html"
        interno.write_text("<html></html>", encoding="utf-8")
        mon.registrar_artefato(interno)
        externo = tmp_path / "clientes.csv"
        externo.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(ValueError):
            mon.registrar_artefato(externo)

    artefatos = {info.path for info in mon.client.list_artifacts(mon.run_id)}
    assert "relatorio.html" in artefatos
    assert "clientes.csv" not in artefatos and "logs.jsonl" in artefatos


def test_aceita_diretorio_relativo_resolvido_no_construtor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mon = Monitoramento(Path("monitoramento"), "exec-relativo")
    outro = tmp_path / "outro"
    outro.mkdir()
    monkeypatch.chdir(outro)  # chdir depois do construtor não pode quebrar o run

    with mon:
        mon.registrar_metricas({"linhas": 7})

    assert mon.diretorio.is_absolute()
    assert mon.diretorio == tmp_path / "monitoramento"
    assert (tmp_path / "monitoramento" / "mlflow.db").exists()  # mesmo diretório
    assert not (outro / "monitoramento").exists()
    experimento = mon.client.get_experiment_by_name("credito-pipeline")
    assert experimento is not None
    assert experimento.artifact_location.startswith("file:")
    assert "mlartifacts" in experimento.artifact_location
    run = mon.client.get_run(mon.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.artifact_uri.startswith("file:")
    artefatos = {info.path for info in mon.client.list_artifacts(mon.run_id)}
    assert "logs.jsonl" in artefatos


def test_metricas_nao_finitas_rejeitadas_antes_de_logar(tmp_path):
    with Monitoramento(tmp_path, "exec-nan") as mon:
        with pytest.raises(ValueError, match="nan"):
            mon.registrar_metricas({"ok": 1.0, "quebrada": float("nan")})
        with pytest.raises(ValueError):
            mon.registrar_metricas({"quebrada": float("inf")})
        with pytest.raises(ValueError):
            mon.registrar_metricas({"quebrada": float("-inf")})
        # validação antes do log: sem batch parcial no MLflow nem no JSONL
        assert mon.client.get_metric_history(mon.run_id, "ok") == []
        assert mon.client.get_metric_history(mon.run_id, "quebrada") == []
        eventos = _ler_logs(mon.diretorio_execucao)
        assert not [evento for evento in eventos if evento["evento"] == "metricas"]

        registradas = mon.registrar_metricas({"finita": 3.5, "opcional": None})
        assert registradas == {"finita": 3.5}  # None continua omitido
        run = mon.client.get_run(mon.run_id)
        assert run.data.metrics["finita"] == 3.5
        assert "opcional" not in run.data.metrics

    texto = (mon.diretorio_execucao / "logs.jsonl").read_text(encoding="utf-8")

    def _recusar_constante(nome):
        raise AssertionError(f"JSONL não é JSON estrito: constante {nome}")

    for linha in texto.splitlines():
        json.loads(linha, parse_constant=_recusar_constante)


def test_falha_persistencia_artefato_marca_erro_real_sem_falso_sucesso(
    tmp_path, monkeypatch
):
    mon = Monitoramento(tmp_path, "exec-upload-falho")

    def upload_quebrado(*_args, **_kwargs):
        raise OSError("falha simulada de upload")

    with pytest.raises(OSError, match="falha simulada"), mon:
        mon.registrar_metricas({"linhas": 5})
        monkeypatch.setattr(mon.client, "log_artifact", upload_quebrado)

    run = mon.client.get_run(mon.run_id)
    assert run.info.status == "FAILED"  # não RUNNING com cara de sucesso
    assert run.data.tags["resultado"] == "erro"
    assert run.data.tags["codigo_saida"] == "2"
    eventos = _ler_logs(mon.diretorio_execucao)
    assert not [evento for evento in eventos if evento["evento"] == "artefato"]
    resultados = [evento for evento in eventos if evento["evento"] == "resultado"]
    assert resultados[-1]["status"] == "erro"  # desfecho final não é "ok"
    erros = [evento for evento in eventos if evento["evento"] == "execucao_erro"]
    assert erros and erros[0]["erro"] == "OSError"
