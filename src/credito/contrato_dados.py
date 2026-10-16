"""Etapa 1 (3/5) — Contrato de dados com Great Expectations.

Define a suite de expectations que todo lote precisa satisfazer para ser
ingerido. As regras são **rígidas** (sem `mostly`): qualquer linha violando
qualquer regra reprova o lote e a ingestão é bloqueada.

Regras obrigatórias do enunciado:
  - idade > 18                  -> `idade_maior_que_18`
  - renda não nula              -> `renda_nao_nula`
  - sem duplicatas              -> `sem_duplicatas`

Regras adicionais de plausibilidade: ver `construir_regras`.

Perfis do contrato:
  - `treino`     (padrão): exige o rótulo `SeriousDlqin2yrs` e exatamente as
    colunas do esquema. É o que vale para a referência.
  - `inferencia`: para lotes de produção. Em crédito o rótulo só aparece meses
    depois (o target mede atraso em 2 anos), então o lote pode chegar sem ele
    e com colunas extras (ex.: as predições). As regras das features são as
    mesmas; só saem a checagem do target e o "match exato" de colunas.

Uso:
    from credito.contrato_dados import validar_dataframe
    result = validar_dataframe(df)                       # perfil treino
    result = validar_dataframe(df, perfil="inferencia")  # lote de produção
    result.success  # bool
"""

from __future__ import annotations

import sys
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe
from great_expectations.core import ExpectationSuite

from credito.esquema import (
    COLUMNS,
    FEATURES,
    MAX_AGE,
    MAX_PAST_DUE,
    MIN_AGE_EXCLUSIVE,
    PAST_DUE_COLUMNS,
    TARGET,
    VALID_TARGET_VALUES,
)

SUITE_NAME = "give_me_some_credit_ingestion_contract"

PERFIL_TREINO = "treino"
PERFIL_INFERENCIA = "inferencia"
PERFIS = (PERFIL_TREINO, PERFIL_INFERENCIA)

# Finitude numérica de verdade: limites = ±sys.float_info.max. Diferente de
# uma regex, reprova "1e999"/"9"*400 (estouram float64) e strings não
# numéricas, que o GX marca como falha ao comparar com os limites.
# O GX ignora nulos nas expectations de valores, então a checagem de nulo
# fica em uma regra própria (`*_nao_nula` / `target_nao_nulo`).
LIMITE_FINITO = sys.float_info.max

# Colunas cujo valor deve ser sempre >= 0.
NON_NEGATIVE_COLUMNS = [
    "MonthlyIncome",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfDependents",
]

def construir_regras(perfil: str = PERFIL_TREINO) -> list[tuple[str, Any]]:
    """Monta as regras do contrato na ordem em que são reportadas.

    As expectations são criadas a cada chamada: o GX não permite adicionar a
    mesma expectation a duas suites (cada uma recebe um `id` próprio).
    """
    if perfil not in PERFIS:
        raise ValueError(f"Perfil desconhecido: {perfil!r} (use um de {PERFIS})")
    treino = perfil == PERFIL_TREINO

    if treino:
        regra_colunas = (
            "colunas_esperadas",
            gxe.ExpectTableColumnsToMatchSet(column_set=COLUMNS, exact_match=True),
        )
    else:
        # exact_match=False: as features precisam estar lá, extras são aceitas.
        regra_colunas = (
            "features_presentes",
            gxe.ExpectTableColumnsToMatchSet(column_set=FEATURES, exact_match=False),
        )

    regras: list[tuple[str, Any]] = [
        regra_colunas,
        (
            "lote_nao_vazio",
            gxe.ExpectTableRowCountToBeBetween(min_value=1),
        ),
        (
            "idade_maior_que_18",
            gxe.ExpectColumnValuesToBeBetween(
                column="age",
                min_value=MIN_AGE_EXCLUSIVE,
                strict_min=True,
                max_value=MAX_AGE,
            ),
        ),
        (
            "renda_nao_nula",
            gxe.ExpectColumnValuesToNotBeNull(column="MonthlyIncome"),
        ),
        (
            "sem_duplicatas",
            gxe.ExpectCompoundColumnsToBeUnique(
                column_list=COLUMNS if treino else FEATURES
            ),
        ),
    ]

    if treino:
        regras.append(
            (
                "target_binario",
                gxe.ExpectColumnValuesToBeInSet(
                    column=TARGET, value_set=VALID_TARGET_VALUES
                ),
            )
        )
        regras.append(
            (
                "target_nao_nulo",
                gxe.ExpectColumnValuesToNotBeNull(column=TARGET),
            )
        )

    for _column in NON_NEGATIVE_COLUMNS:
        regras.append(
            (
                f"{_column}_nao_negativo",
                gxe.ExpectColumnValuesToBeBetween(column=_column, min_value=0),
            )
        )

    for _column in PAST_DUE_COLUMNS:
        regras.append(
            (
                f"{_column}_em_faixa_plausivel",
                gxe.ExpectColumnValuesToBeBetween(
                    column=_column, min_value=0, max_value=MAX_PAST_DUE
                ),
            )
        )

    # Sanidade de todas as features (os dois perfis): não nula e finita.
    # MonthlyIncome já tem a regra rígida `renda_nao_nula`.
    for _column in FEATURES:
        if _column != "MonthlyIncome":
            regras.append(
                (
                    f"{_column}_nao_nula",
                    gxe.ExpectColumnValuesToNotBeNull(column=_column),
                )
            )
        regras.append(
            (
                f"{_column}_finita_e_numerica",
                gxe.ExpectColumnValuesToBeBetween(
                    column=_column,
                    min_value=-LIMITE_FINITO,
                    max_value=LIMITE_FINITO,
                ),
            )
        )

    return regras


def _rotular(expectation: Any, label: str) -> Any:
    """Marca a expectation com o rótulo da regra em `meta.rule`."""
    expectation.meta = {"rule": label}
    return expectation


def criar_contexto() -> gx.DataContext:
    """Contexto efêmero do GX (sem arquivos de configuração persistidos)."""
    context = gx.get_context(mode="ephemeral")
    # Silencia as barras de progresso do cálculo de métricas (saída limpa no CI).
    context.variables.progress_bars = {"globally": False}
    return context


def criar_suite(
    context: gx.DataContext, perfil: str = PERFIL_TREINO
) -> ExpectationSuite:
    """Monta a suite do contrato, com o rótulo de cada regra em `meta.rule`."""
    suite = context.suites.add(ExpectationSuite(name=f"{SUITE_NAME}_{perfil}"))
    for label, expectation in construir_regras(perfil):
        suite.add_expectation(_rotular(expectation, label))
    return suite


def validar_dataframe(frame: pd.DataFrame, perfil: str = PERFIL_TREINO) -> Any:
    """Valida um DataFrame contra o contrato e devolve o `ValidationResult` do GX."""
    context = criar_contexto()
    suite = criar_suite(context, perfil)

    data_source = context.data_sources.add_pandas(name="ingestion_source")
    data_asset = data_source.add_dataframe_asset(name="ingestion_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "ingestion_batch"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})

    return batch.validate(suite)
