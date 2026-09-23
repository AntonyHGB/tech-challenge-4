"""Etapa 2: lotes sintéticos, sem Kaggle ou rede."""

from __future__ import annotations

import json
import sys

import pytest
from conftest import construir_lote

from credito.drift import (
    EXIT_DRIFT,
    EXIT_ERROR,
    EXIT_INVALID,
    EXIT_STABLE,
    comparar,
    main,
)


def _arquivos(tmp_path, current, reference=None):
    reference_path = tmp_path / "referencia.csv"
    current_path = tmp_path / "lote.csv"
    report_path = tmp_path / "relatorio.json"
    (reference if reference is not None else construir_lote(300)).to_csv(
        reference_path, index=False
    )
    current.to_csv(current_path, index=False)
    return reference_path, current_path, report_path


def test_lote_estavel_nao_alerta(tmp_path):
    reference = construir_lote(300)
    paths = _arquivos(tmp_path, reference.copy(), reference)
    assert comparar(*paths) == EXIT_STABLE
    report = json.loads(paths[2].read_text())
    assert report["status"] == "sem_drift"
    assert report["agregado"]["colunas_com_drift"] == 0
    assert len(report["colunas"]) == 10
    assert report["referencia"]["sha256"] == report["lote"]["sha256"]


def test_deslocamento_valido_alerta(tmp_path):
    reference = construir_lote(300)
    current = reference.copy()
    current["age"] += 20
    current["MonthlyIncome"] *= 3
    current["DebtRatio"] += 10
    current["RevolvingUtilizationOfUnsecuredLines"] += 2
    current["NumberOfDependents"] += 5
    current["NumberOfTimes90DaysLate"] += 5
    paths = _arquivos(tmp_path, current, reference)
    assert comparar(*paths) == EXIT_DRIFT
    report = json.loads(paths[2].read_text())
    assert report["status"] == "drift_detectado"
    assert report["agregado"]["colunas_com_drift"] >= 5
    assert report["agregado"]["share"] >= 0.5
    assert report["configuracao"]["metodo"] == "ks"


def test_contrato_invalido_nao_e_drift(tmp_path):
    current = construir_lote(300)
    current.loc[0, "age"] = 16
    paths = _arquivos(tmp_path, current)
    assert comparar(*paths) == EXIT_INVALID
    report = json.loads(paths[2].read_text())
    assert report["status"] == "lote_invalido"
    assert "idade_maior_que_18" in report["regras_violadas"]
    assert "agregado" not in report


def test_amostra_insuficiente_e_erro_distinto(tmp_path):
    current = construir_lote(60)
    paths = _arquivos(tmp_path, current)
    assert comparar(*paths) == EXIT_ERROR
    report = json.loads(paths[2].read_text())
    assert report["status"] == "amostra_insuficiente"
    assert report["lote"]["linhas"] == 60
    assert "agregado" not in report


def test_cli_sucesso_substitui_relatorio_e_usa_caminho_relativo(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    paths = _arquivos(tmp_path, construir_lote(101))
    paths[2].write_text('{"status": "drift_detectado"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            "lote.csv",
            "--reference",
            "referencia.csv",
            "--report",
            "relatorio.json",
        ],
    )

    assert main() == EXIT_STABLE
    assert "exit code 0" in capsys.readouterr().out
    report = json.loads(paths[2].read_text())
    assert report["status"] == "sem_drift"
    assert report["referencia"]["arquivo"] == "referencia.csv"
    assert report["lote"]["arquivo"] == "lote.csv"


def test_referencia_invalida_remove_relatorio_anterior(tmp_path, monkeypatch, capsys):
    reference = construir_lote(101)
    reference.loc[0, "age"] = 16
    paths = _arquivos(tmp_path, construir_lote(101), reference)
    paths[2].write_text('{"status": "sem_drift"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            str(paths[1]),
            "--reference",
            str(paths[0]),
            "--report",
            str(paths[2]),
        ],
    )

    assert main() == EXIT_ERROR
    assert "Referência não passou" in capsys.readouterr().err
    assert not paths[2].exists()


def test_erro_leitura_remove_relatorio_anterior(tmp_path, monkeypatch, capsys):
    paths = _arquivos(tmp_path, construir_lote(101))
    paths[1].unlink()
    paths[2].write_text('{"status": "sem_drift"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            str(paths[1]),
            "--reference",
            str(paths[0]),
            "--report",
            str(paths[2]),
        ],
    )

    assert main() == EXIT_ERROR
    assert "FileNotFoundError" in capsys.readouterr().err
    assert not paths[2].exists()


def test_erro_escrita_remove_relatorio_anterior(tmp_path, monkeypatch, capsys):
    paths = _arquivos(tmp_path, construir_lote(101))
    paths[2].write_text('{"status": "sem_drift"}')

    def falhar(*args):
        raise OSError("falha simulada na gravação")

    monkeypatch.setattr("credito.drift.os.replace", falhar)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            str(paths[1]),
            "--reference",
            str(paths[0]),
            "--report",
            str(paths[2]),
        ],
    )
    assert main() == EXIT_ERROR
    assert "falha simulada" in capsys.readouterr().err
    assert not paths[2].exists()


@pytest.mark.parametrize("csv_index", [0, 1])
def test_relatorio_nao_sobrescreve_csv_mesmo_via_symlink(
    tmp_path, monkeypatch, capsys, csv_index
):
    paths = _arquivos(tmp_path, construir_lote(101))
    alias = tmp_path / "alias.json"
    alias.symlink_to(paths[csv_index])
    original = paths[csv_index].read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            str(paths[1]),
            "--reference",
            str(paths[0]),
            "--report",
            str(alias),
        ],
    )

    assert main() == EXIT_ERROR
    assert "diferente dos CSVs" in capsys.readouterr().err
    assert paths[csv_index].read_bytes() == original


def test_arquivo_fora_do_cwd_nao_expoe_diretorios(tmp_path, monkeypatch):
    paths = _arquivos(tmp_path, construir_lote(101))
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    assert comparar(*paths) == EXIT_STABLE
    report = json.loads(paths[2].read_text())
    assert report["referencia"]["arquivo"] == "referencia.csv"
