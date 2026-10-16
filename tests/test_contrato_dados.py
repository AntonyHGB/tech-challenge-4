"""Testes do contrato de dados: o lote limpo passa e cada defeito reprova."""

from __future__ import annotations

import numpy as np
import pytest

from credito.contrato_dados import construir_regras, validar_dataframe
from credito.esquema import FEATURES, TARGET

# --- Rótulos originais do contrato (precisam ser preservados) ---
ROTULOS_TREINO_ORIGINAIS = [
    "colunas_esperadas",
    "lote_nao_vazio",
    "idade_maior_que_18",
    "renda_nao_nula",
    "sem_duplicatas",
    "target_binario",
    "MonthlyIncome_nao_negativo",
    "RevolvingUtilizationOfUnsecuredLines_nao_negativo",
    "DebtRatio_nao_negativo",
    "NumberOfDependents_nao_negativo",
    "NumberOfTime30-59DaysPastDueNotWorse_em_faixa_plausivel",
    "NumberOfTimes90DaysLate_em_faixa_plausivel",
    "NumberOfTime60-89DaysPastDueNotWorse_em_faixa_plausivel",
]
ROTULOS_INFERENCIA_ORIGINAIS = [
    rotulo
    for rotulo in ROTULOS_TREINO_ORIGINAIS
    if rotulo not in ("target_binario", "colunas_esperadas")
] + ["features_presentes"]

# --- Sanidade numérica: nulo, infinito e não numérico ---
# MonthlyIncome já é coberto pela regra rígida renda_nao_nula; o nulo do
# alvo tem teste próprio (test_target_nulo_reprova_no_treino).
ROTULOS_NULOS = [
    f"{coluna}_nao_nula" for coluna in FEATURES if coluna != "MonthlyIncome"
] + ["renda_nao_nula"]
ROTULOS_FINITAS = [f"{coluna}_finita_e_numerica" for coluna in FEATURES]

TOTAL_DE_REGRAS = 33  # perfil treino
TOTAL_INFERENCIA = 31  # treino sem target_binario e target_nao_nulo


def _regras_violadas(resultado) -> list[str]:
    """Rótulos das regras reprovadas, em ordem alfabética."""
    return sorted(
        detail.expectation_config.meta["rule"]
        for detail in resultado.results
        if not detail.success
    )


def _corromper(lote_limpo, colunas: list[str], valor):
    """Copia o lote e injeta `valor` na primeira linha de `colunas`."""
    lote = lote_limpo.copy()
    if isinstance(valor, str):
        lote[colunas] = lote[colunas].astype(object)
    else:
        lote[colunas] = lote[colunas].astype(float)
    lote.loc[lote.index[0], colunas] = valor
    return lote


def test_lote_limpo_passa_no_contrato(lote_limpo):
    """Lote sem defeitos aprova em todas as regras do contrato."""
    resultado = validar_dataframe(lote_limpo)
    assert resultado.success, _regras_violadas(resultado)
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


# --- Perfil de inferência (lote de produção, rótulo ainda não chegou) ---


def test_treino_reprova_lote_sem_rotulo(lote_limpo):
    lote = lote_limpo.drop(columns=["SeriousDlqin2yrs"])
    violadas = _regras_violadas(validar_dataframe(lote))
    assert "colunas_esperadas" in violadas


def test_inferencia_aceita_lote_sem_rotulo(lote_limpo):
    lote = lote_limpo.drop(columns=["SeriousDlqin2yrs"])
    resultado = validar_dataframe(lote, perfil="inferencia")
    assert resultado.success, _regras_violadas(resultado)
    assert len(resultado.results) == TOTAL_INFERENCIA  # sem regras de target


def test_inferencia_aceita_colunas_de_predicao(lote_limpo):
    lote = lote_limpo.copy()
    lote["probabilidade_inadimplencia"] = 0.3
    lote["classe_prevista"] = 0
    assert validar_dataframe(lote, perfil="inferencia").success


def test_inferencia_continua_barrando_dado_ruim(lote_limpo):
    lote = lote_limpo.drop(columns=["SeriousDlqin2yrs"])
    lote.loc[lote.index[0], "age"] = 16
    lote.loc[lote.index[1], "MonthlyIncome"] = np.nan
    violadas = _regras_violadas(validar_dataframe(lote, perfil="inferencia"))
    assert violadas == ["idade_maior_que_18", "renda_nao_nula"]


def test_inferencia_reprova_se_faltar_feature(lote_limpo):
    lote = lote_limpo.drop(columns=["DebtRatio"])
    violadas = _regras_violadas(validar_dataframe(lote, perfil="inferencia"))
    assert "features_presentes" in violadas


def test_perfil_desconhecido_da_erro(lote_limpo):
    with pytest.raises(ValueError, match="Perfil desconhecido"):
        validar_dataframe(lote_limpo, perfil="producao")


# --- Sanidade: nulo, infinito e não numérico em TODAS as features ---


@pytest.mark.parametrize("perfil", ["treino", "inferencia"])
def test_features_nulas_reprovam_em_todos_os_perfis(lote_limpo, perfil):
    """NaN/nulo em qualquer feature reprova (rótulo por feature)."""
    lote = _corromper(lote_limpo, FEATURES, np.nan)
    violadas = set(_regras_violadas(validar_dataframe(lote, perfil=perfil)))
    faltantes = set(ROTULOS_NULOS) - violadas
    assert not faltantes, f"regras de nulo ausentes: {sorted(faltantes)}"


@pytest.mark.parametrize("perfil", ["treino", "inferencia"])
@pytest.mark.parametrize("valor", [np.inf, -np.inf], ids=["inf", "-inf"])
def test_features_infinitas_reprovam_em_todos_os_perfis(lote_limpo, perfil, valor):
    """+inf/-inf em qualquer feature reprova (rótulo por feature)."""
    lote = _corromper(lote_limpo, FEATURES, valor)
    violadas = set(_regras_violadas(validar_dataframe(lote, perfil=perfil)))
    faltantes = set(ROTULOS_FINITAS) - violadas
    assert not faltantes, f"regras de finitez ausentes: {sorted(faltantes)}"


@pytest.mark.parametrize("perfil", ["treino", "inferencia"])
def test_features_nao_numericas_reprovam_em_todos_os_perfis(lote_limpo, perfil):
    """Texto em qualquer feature reprova (rótulo por feature)."""
    lote = _corromper(lote_limpo, FEATURES, "abc")
    violadas = set(_regras_violadas(validar_dataframe(lote, perfil=perfil)))
    faltantes = set(ROTULOS_FINITAS) - violadas
    assert not faltantes, f"regras numéricas ausentes: {sorted(faltantes)}"


@pytest.mark.parametrize(
    "valor",
    ["1e999", "9" * 400],
    ids=["overflow_scientifico", "cadeia_longa_de_digitos"],
)
@pytest.mark.parametrize("perfil", ["treino", "inferencia"])
def test_feature_object_com_string_numerica_ilegivel_reprova(
    lote_limpo, perfil, valor
):
    """String que 'parece' número ('1e999') precisa reprovar por finitude real.

    Uma regex aceita '1e999'/'9'*400, mas tais valores estouram float64
    (viram inf ou não são numeric dtype); a regra precisa rejeitá-los.
    """
    coluna = "NumberOfOpenCreditLinesAndLoans"
    lote = _corromper(lote_limpo, [coluna], valor)
    violadas = _regras_violadas(validar_dataframe(lote, perfil=perfil))
    assert f"{coluna}_finita_e_numerica" in violadas


def test_target_nulo_reprova_no_treino(lote_limpo):
    """Target nulo reprova no perfil treino (além de fora de {0,1})."""
    lote = _corromper(lote_limpo, [TARGET], np.nan)
    violadas = _regras_violadas(validar_dataframe(lote))
    assert "target_nao_nulo" in violadas


def test_rotulos_antigos_preservados_e_contagens():
    """Os rótulos originais seguem na suite e as contagens batem."""
    treino = [rotulo for rotulo, _ in construir_regras("treino")]
    inferencia = [rotulo for rotulo, _ in construir_regras("inferencia")]
    assert set(ROTULOS_TREINO_ORIGINAIS) <= set(treino)
    assert set(ROTULOS_INFERENCIA_ORIGINAIS) <= set(inferencia)
    assert len(treino) == TOTAL_DE_REGRAS
    assert len(inferencia) == TOTAL_INFERENCIA
