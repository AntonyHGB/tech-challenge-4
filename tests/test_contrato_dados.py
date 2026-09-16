"""Testes do contrato de dados: o lote limpo passa e cada defeito reprova."""

from __future__ import annotations

import numpy as np
import pytest

from credito.contrato_dados import validar_dataframe

TOTAL_DE_REGRAS = 13


def _regras_violadas(resultado) -> list[str]:
    """Rótulos das regras reprovadas, em ordem alfabética."""
    return sorted(
        detail.expectation_config.meta["rule"]
        for detail in resultado.results
        if not detail.success
    )


def test_lote_limpo_passa_no_contrato(lote_limpo):
    """Lote sem defeitos aprova nas 13 regras."""
    resultado = validar_dataframe(lote_limpo)
    assert resultado.success
    assert len(resultado.results) == TOTAL_DE_REGRAS
    assert _regras_violadas(resultado) == []


@pytest.mark.parametrize("idade", [17, 18])
def test_idade_menor_ou_igual_a_18_reprova(lote_limpo, idade):
    """Idade <= 18 viola a regra rígida idade_maior_que_18."""
    lote = lote_limpo.copy()
    lote.loc[lote.index[0], "age"] = idade
    assert _regras_violadas(validar_dataframe(lote)) == ["idade_maior_que_18"]


def test_renda_nula_reprova(lote_limpo):
    """MonthlyIncome nula viola a regra rígida renda_nao_nula."""
    lote = lote_limpo.copy()
    lote.loc[lote.index[0], "MonthlyIncome"] = np.nan
    assert _regras_violadas(validar_dataframe(lote)) == ["renda_nao_nula"]


def test_linha_duplicada_reprova(lote_limpo):
    """Linha repetida viola a regra rígida sem_duplicatas."""
    lote = lote_limpo.copy()
    lote.loc[lote.index[1]] = lote.loc[lote.index[0]]
    assert _regras_violadas(validar_dataframe(lote)) == ["sem_duplicatas"]


def test_target_fora_de_zero_e_um_reprova(lote_limpo):
    """Target fora de {0, 1} viola a regra target_binario."""
    lote = lote_limpo.copy()
    lote.loc[lote.index[0], "SeriousDlqin2yrs"] = 2
    assert _regras_violadas(validar_dataframe(lote)) == ["target_binario"]
