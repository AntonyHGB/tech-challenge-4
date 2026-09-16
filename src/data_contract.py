"""Etapa 1 (3/5) — Contrato de dados com Great Expectations.

Define a suite de expectations que todo lote precisa satisfazer para ser
ingerido. As regras são **rígidas** (sem `mostly`): qualquer linha violando
qualquer regra reprova o lote e a ingestão é bloqueada.

Regras obrigatórias do enunciado:
  - idade > 18                  -> `idade_maior_que_18`
  - renda não nula              -> `renda_nao_nula`
  - sem duplicatas              -> `sem_duplicatas`

Regras adicionais de plausibilidade: ver `contract_rules()`.

Uso:
    from data_contract import validate_dataframe
    result = validate_dataframe(df)
    result.success  # bool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations import expectations as gxe
from great_expectations.core import ExpectationSuite
from schema import (
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


@dataclass(frozen=True)
class ContractRule:
    """Uma regra do contrato, com rótulo legível para relatórios."""

    label: str
    expectation: Any


def contract_rules() -> list[ContractRule]:
    """Regras do contrato na ordem em que são reportadas."""
    rules = [
        ContractRule(
            "colunas_esperadas",
            gxe.ExpectTableColumnsToMatchSet(column_set=COLUMNS, exact_match=True),
        ),
        ContractRule(
            "lote_nao_vazio",
            gxe.ExpectTableRowCountToBeBetween(min_value=1),
        ),
        ContractRule(
            "idade_maior_que_18",
            gxe.ExpectColumnValuesToBeBetween(
                column="age",
                min_value=MIN_AGE_EXCLUSIVE,
                strict_min=True,
                max_value=MAX_AGE,
            ),
        ),
        ContractRule(
            "renda_nao_nula",
            gxe.ExpectColumnValuesToNotBeNull(column="MonthlyIncome"),
        ),
        ContractRule(
            "sem_duplicatas",
            gxe.ExpectCompoundColumnsToBeUnique(column_list=COLUMNS),
        ),
        ContractRule(
            "target_binario",
            gxe.ExpectColumnValuesToBeInSet(column=TARGET, value_set=VALID_TARGET_VALUES),
        ),
    ]

    for column in NON_NEGATIVE_COLUMNS:
        rules.append(
            ContractRule(
                f"{column}_nao_negativo",
                gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0),
            )
        )

    for column in PAST_DUE_COLUMNS:
        rules.append(
            ContractRule(
                f"{column}_em_faixa_plausivel",
                gxe.ExpectColumnValuesToBeBetween(
                    column=column, min_value=0, max_value=MAX_PAST_DUE
                ),
            )
        )

    return rules


def build_context() -> gx.DataContext:
    """Contexto efêmero do GX (sem arquivos de configuração persistidos)."""
    context = gx.get_context(mode="ephemeral")
    # Silencia as barras de progresso do cálculo de métricas (saída limpa no CI).
    context.variables.progress_bars = {"globally": False}
    return context


def build_suite(context: gx.DataContext) -> ExpectationSuite:
    """Monta a suite do contrato, com o rótulo de cada regra em `meta.rule`."""
    suite = context.suites.add(ExpectationSuite(name=SUITE_NAME))
    for rule in contract_rules():
        rule.expectation.meta = {"rule": rule.label}
        suite.add_expectation(rule.expectation)
    return suite


def validate_dataframe(
    frame: pd.DataFrame, context: gx.DataContext | None = None
) -> Any:
    """Valida um DataFrame contra o contrato e devolve o `ValidationResult` do GX."""
    context = context or build_context()
    suite = build_suite(context)

    data_source = context.data_sources.add_pandas(name="ingestion_source")
    data_asset = data_source.add_dataframe_asset(name="ingestion_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("ingestion_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": frame})

    return batch.validate(suite)


def load_batch(csv_path: str | Any) -> pd.DataFrame:
    """Carrega um CSV de lote para validação."""
    return pd.read_csv(csv_path)
