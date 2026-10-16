"""Etapa 2: lotes sintéticos, sem Kaggle ou rede."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import construir_lote

from credito.drift import (
    EXIT_DRIFT,
    EXIT_ERROR,
    EXIT_INVALID,
    EXIT_STABLE,
    PSI_FORTE,
    calcular_psi,
    comparar,
    main,
    nivel_psi,
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
            "--html",
            "relatorio.html",
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
            "--html",
            str(paths[2].with_suffix(".html")),
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
            "--html",
            str(paths[2].with_suffix(".html")),
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
            "--html",
            str(paths[2].with_suffix(".html")),
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
    try:
        alias.symlink_to(paths[csv_index])
    except OSError as exc:
        # Somente WinError 1314 (sem privilégio de symlink no Windows):
        # qualquer outro erro de E/S continua falhando o teste (não mascarar).
        if getattr(exc, "winerror", None) != 1314:
            raise
        pytest.skip(
            "WinError 1314: Windows sem privilégio/Developer Mode para criar "
            "symlinks; teste de portabilidade ignorado (roda normalmente no Linux)"
        )
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


# --- PSI e perfil de inferência no drift ---


def test_psi_zero_para_distribuicao_igual():
    lote = construir_lote(500)
    assert calcular_psi(lote["MonthlyIncome"], lote["MonthlyIncome"]) == 0.0
    assert nivel_psi(0.0) == "estavel"


def test_psi_forte_para_distribuicao_deslocada():
    lote = construir_lote(500)
    psi = calcular_psi(lote["MonthlyIncome"], lote["MonthlyIncome"] * 2)
    assert psi >= 0.25
    assert nivel_psi(psi) == "forte"


def test_psi_funciona_em_coluna_de_contagem_com_muitos_zeros():
    """Atrasos: quase tudo é 0, os quantis se repetem e mesmo assim mede o shift."""
    referencia = construir_lote(500)["NumberOfTimes90DaysLate"] * 0
    referencia.iloc[:25] = 1
    deslocada = referencia + 1
    assert calcular_psi(referencia, deslocada) >= 0.25


def test_json_tem_psi_por_coluna(tmp_path):
    reference = construir_lote(300)
    current = reference.copy()
    current["MonthlyIncome"] *= 3
    paths = _arquivos(tmp_path, current, reference)
    comparar(*paths)
    report = json.loads(paths[2].read_text())
    assert report["colunas"]["MonthlyIncome"]["psi_nivel"] == "forte"
    assert report["colunas"]["age"]["psi"] == 0.0


def test_lote_de_producao_sem_rotulo_e_com_predicoes_e_aceito(tmp_path):
    """Perfil inferencia: o drift roda mesmo sem o target e com colunas extras."""
    reference = construir_lote(300)
    current = reference.drop(columns=["SeriousDlqin2yrs"])
    current["probabilidade_inadimplencia"] = 0.2
    current["classe_prevista"] = 0
    paths = _arquivos(tmp_path, current, reference)
    assert comparar(*paths) == EXIT_STABLE


# --- Robustez da API de PSI: NaN/inf/vazio/tipo/bins viram ValueError ---


def test_psi_rejeita_nan_e_inf_em_qualquer_lado():
    """Um valor inválido basta: nada de filtrar a série em silêncio."""
    base = pd.Series([1.0, 2.0, 3.0] * 10)
    invalidas = (
        pd.Series([1.0, np.nan, 3.0] * 10),
        pd.Series([1.0, np.inf, 3.0] * 10),
        pd.Series([1.0, -np.inf, 3.0] * 10),
    )
    for quebrada in invalidas:
        with pytest.raises(ValueError, match="fin"):
            calcular_psi(quebrada, base)
        with pytest.raises(ValueError, match="fin"):
            calcular_psi(base, quebrada)


def test_psi_rejeita_serie_vazia():
    base = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="vazia"):
        calcular_psi(pd.Series(dtype="float64"), base)
    with pytest.raises(ValueError, match="vazia"):
        calcular_psi(base, pd.Series(dtype="float64"))


def test_psi_rejeita_entrada_bidimensional():
    base = pd.Series([1.0, 2.0] * 50)
    with pytest.raises(ValueError, match="unidimensional"):
        calcular_psi(pd.DataFrame({"a": base, "b": base}), base)
    with pytest.raises(ValueError, match="unidimensional"):
        calcular_psi(base, np.column_stack([base, base]))


def test_psi_rejeita_coluna_nao_numerica():
    base = pd.Series([1.0, 2.0] * 50)
    texto = pd.Series(["azul", "verde"] * 50)
    with pytest.raises(ValueError, match="num"):
        calcular_psi(texto, base)
    with pytest.raises(ValueError, match="num"):
        calcular_psi(base, texto)


@pytest.mark.parametrize("bins", [0, 1, -3, 2.5, "10", None])
def test_psi_rejeita_bins_invalido(bins):
    """bins < 2 deixa faixas degeneradas: PSI daria 0 falso para qualquer coisa."""
    base = pd.Series([1.0, 2.0] * 50)
    with pytest.raises(ValueError, match="bins"):
        calcular_psi(base, base, bins=bins)


# --- Referência constante: shift para cima E para baixo ---


def test_psi_referencia_constante_detecta_shift_para_cima_e_para_baixo():
    """Sem faixa própria para o valor da borda, o shift p/ baixo fica no mesmo bin."""
    referencia = pd.Series([5.0] * 200)
    assert calcular_psi(referencia, referencia + 1.0) >= PSI_FORTE
    assert calcular_psi(referencia, referencia - 1.0) >= PSI_FORTE


def test_psi_referencia_constante_identica_continua_zero():
    referencia = pd.Series([5.0] * 200)
    assert calcular_psi(referencia, referencia) == 0.0
    assert nivel_psi(0.0) == "estavel"


def test_psi_referencia_constante_simetrico_nos_dois_sentidos():
    referencia = pd.Series([5.0] * 200)
    subida = calcular_psi(referencia, referencia + 1.0)
    descida = calcular_psi(referencia, referencia - 1.0)
    assert subida == pytest.approx(descida, rel=1e-12)


def test_psi_normaliza_massas_depois_de_suavizar():
    """Faixa vazia ganha epsilon e as massas seguem somando 1 após o clip."""
    referencia = pd.Series([5.0] * 200)
    atual = pd.Series([3.0] * 100 + [5.0] * 100)
    massa_ref = np.array([0.0, 1.0, 0.0])  # faixas: <5, ==5, >5
    massa_atual = np.array([0.5, 0.5, 0.0])
    suave_ref = np.clip(massa_ref, 1e-6, None)
    suave_atual = np.clip(massa_atual, 1e-6, None)
    suave_ref = suave_ref / suave_ref.sum()
    suave_atual = suave_atual / suave_atual.sum()
    esperado = float(
        np.sum((suave_atual - suave_ref) * np.log(suave_atual / suave_ref))
    )
    assert calcular_psi(referencia, atual) == pytest.approx(esperado, rel=1e-9)


# --- Proteção dos destinos: --html nunca pode colidir com --report ---


def test_html_igual_ao_relatorio_retorna_erro_sem_apagar(
    tmp_path, monkeypatch, capsys
):
    paths = _arquivos(tmp_path, construir_lote(101))
    destino = tmp_path / "saida.json"
    destino.write_text('{"status": "anterior"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detectar_drift.py",
            str(paths[1]),
            "--reference",
            str(paths[0]),
            "--report",
            str(destino),
            "--html",
            str(destino),
        ],
    )

    assert main() == EXIT_ERROR
    assert "diferente do relatório" in capsys.readouterr().err
    assert destino.read_text() == '{"status": "anterior"}'


def test_html_alias_normalizado_do_relatorio_nao_apaga(tmp_path):
    paths = _arquivos(tmp_path, construir_lote(101))
    destino = tmp_path / "saida.json"
    destino.write_text('{"status": "anterior"}')
    (tmp_path / "sub").mkdir()
    # Mesmo arquivo escritório a partir de outro caminho, sem symlink
    # (no Windows exigiria privilégio — WinError 1314).
    alias = tmp_path / "sub" / ".." / "saida.json"

    with pytest.raises(ValueError, match="diferente do relatório"):
        comparar(paths[0], paths[1], destino, html_path=alias)
    assert destino.read_text() == '{"status": "anterior"}'


# --- Integração: lote inválido sem rótulo não gera HTML ---


def test_feature_invalida_sem_rotulo_e_lote_invalido_sem_html(tmp_path):
    """Inferência aceita lote sem rótulo; feature inválida é EXIT_INVALID.

    Nada de HTML novo (nem o obsoleto sobrevive) quando o lote é reprovado.
    """
    reference = construir_lote(300)
    current = reference.drop(columns=["SeriousDlqin2yrs"])
    current.loc[0, "age"] = np.nan
    paths = _arquivos(tmp_path, current, reference)
    html_path = tmp_path / "relatorio.html"
    html_path.write_text("<html>obsoleto</html>")

    code = comparar(*paths, html_path=html_path)
    assert code == EXIT_INVALID
    report = json.loads(paths[2].read_text())
    assert report["status"] == "lote_invalido"
    assert "agregado" not in report
    assert any("age" in rule for rule in report["regras_violadas"])
    assert not html_path.exists()
