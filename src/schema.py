"""Fonte única de verdade do esquema do dataset Give Me Some Credit.

Todos os módulos (download, preparação, contrato, treino, validação)
importam daqui para evitar divergência de nomes/regras.
"""

from __future__ import annotations

# Coluna de índice gerada pelo CSV original do Kaggle (descartada na limpeza).
RAW_INDEX_COLUMN = "Unnamed: 0"

TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

COLUMNS = [TARGET] + FEATURES

# Contadores de atraso ("past due"). No arquivo original do Kaggle eles usam
# 96 e 98 como códigos de sentinela, que não são contagens reais de atrasos.
PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
PAST_DUE_SENTINELS = (96, 98)

# Colunas com valores ausentes no arquivo original (tratadas na referência).
NULLABLE_COLUMNS = ["MonthlyIncome", "NumberOfDependents"]

# Regras de plausibilidade usadas pelo contrato de dados.
MIN_AGE_EXCLUSIVE = 18
MAX_AGE = 120
MAX_PAST_DUE = 50
VALID_TARGET_VALUES = [0, 1]
