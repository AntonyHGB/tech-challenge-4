"""Testes dos exit codes do validador, executado via subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from credito.contrato_dados import SUITE_NAME, construir_regras

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "validar_lote.py"


def _rodar_validador(
    csv: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(csv), *args],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )


def test_lote_aprovado_sai_com_exit_0(lote_limpo, tmp_path):
    """Lote no contrato: PASS, todas as regras e exit code 0 (liberada)."""
    csv = tmp_path / "referencia.csv"
    lote_limpo.to_csv(csv, index=False)

    processo = _rodar_validador(csv)
    total = len(construir_regras("treino"))

    assert processo.returncode == 0
    assert f"Regras: {total}/{total} aprovadas" in processo.stdout
    assert "ingestão LIBERADA" in processo.stdout
    # A suite real usa o sufixo de perfil; o print precisa mostrá-lo.
    assert f"{SUITE_NAME}_treino (perfil: treino)" in processo.stdout


def test_lote_reprovado_sai_com_exit_1(lote_limpo, tmp_path):
    """Lote fora do contrato: FAIL e exit code 1 (ingestão bloqueada)."""
    lote = lote_limpo.copy()
    lote.loc[lote.index[0], "age"] = 16
    lote.loc[lote.index[1], "MonthlyIncome"] = float("nan")
    csv = tmp_path / "corrompido.csv"
    lote.to_csv(csv, index=False)

    processo = _rodar_validador(csv)

    assert processo.returncode == 1
    assert "ingestão BLOQUEADA" in processo.stdout
    assert "idade_maior_que_18" in processo.stdout
    assert "renda_nao_nula" in processo.stdout


def test_arquivo_ausente_sai_com_exit_2(tmp_path):
    """Arquivo inexistente: exit code 2 (erro de execução)."""
    processo = _rodar_validador(tmp_path / "nao_existe.csv")

    assert processo.returncode == 2
    assert "Arquivo não encontrado" in processo.stderr


def test_perfil_inferencia_aprova_lote_sem_rotulo(lote_limpo, tmp_path):
    """Lote de produção sem rótulo: reprova no perfil treino, passa no inferencia."""
    csv = tmp_path / "producao.csv"
    lote_limpo.drop(columns=["SeriousDlqin2yrs"]).to_csv(csv, index=False)

    assert _rodar_validador(csv).returncode == 1

    processo = _rodar_validador(csv, "--perfil", "inferencia")
    total = len(construir_regras("inferencia"))

    assert processo.returncode == 0
    assert "perfil: inferencia" in processo.stdout
    assert f"{SUITE_NAME}_inferencia (perfil: inferencia)" in processo.stdout
    assert f"Regras: {total}/{total} aprovadas" in processo.stdout
