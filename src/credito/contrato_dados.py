"""Etapa 1 (3/5) — Contrato de dados com Great Expectations.

Define a suite de expectations que todo lote precisa satisfazer para ser
ingerido. As regras são **rígidas** (sem `mostly`): qualquer linha violando
qualquer regra reprova o lote e a ingestão é bloqueada.

Regras obrigatórias do enunciado:
  - idade > 18                  -> `idade_maior_que_18`
  - renda não nula              -> `renda_nao_nula`
  - sem duplicatas              -> `sem_duplicatas`

Regras adicionais de plausibilidade: ver `construir_regras`.

Uso:
    from credito.contrato_dados import validar_dataframe
    result = validar_dataframe(df)
    result.success  # bool
"""

from __future__ import annotations

from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe
from great_expectations.core import ExpectationSuite

from credito.esquema import (
    COLUMNS,
    MAX_AGE,
    MAX_PAST_DUE,
    MIN_AGE_EXCLUSIVE,
    PAST_DUE_COLUMNS,
    TARGET,
    VALID_TARGET_VALUES,
)

SUITE_NAME = "give_me_some_credit_ingestion_contract"

# Colunas cujo valor deve ser sempre >= 0.
NON_NEGATIVE_COLUMNS = [
    "MonthlyIncome",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfDependents",
]

def construir_regras() -> list[tuple[str, Any]]:
    """Monta as regras do contrato na ordem em que são reportadas.

    As expectations são criadas a cada chamada: o GX não permite adicionar a
    mesma expectation a duas suites (cada uma recebe um `id` próprio).
    """
    regras: list[tuple[str, Any]] = [
        (
            "colunas_esperadas",
            gxe.ExpectTableColumnsToMatchSet(column_set=COLUMNS, exact_match=True),
        ),
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
            gxe.ExpectCompoundColumnsToBeUnique(column_list=COLUMNS),
        ),
        (
            "target_binario",
            gxe.ExpectColumnValuesToBeInSet(
                column=TARGET, value_set=VALID_TARGET_VALUES
            ),
        ),
    ]

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


def criar_suite(context: gx.DataContext) -> ExpectationSuite:
    """Monta a suite do contrato, com o rótulo de cada regra em `meta.rule`."""
    suite = context.suites.add(ExpectationSuite(name=SUITE_NAME))
    for label, expectation in construir_regras():
        suite.add_expectation(_rotular(expectation, label))
    return suite


def validar_dataframe(frame: pd.DataFrame) -> Any:
    """Valida um DataFrame contra o contrato e devolve o `ValidationResult` do GX."""
    context = criar_contexto()
    suite = criar_suite(context)

    data_source = context.data_sources.add_pandas(name="ingestion_source")
    data_asset = data_source.add_dataframe_asset(name="ingestion_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "ingestion_batch"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})

    return batch.validate(suite)
