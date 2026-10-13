"""Etapa 2 (1/3) — Dataset de Produção simulado (passagem do tempo).

Parte do dataset de REFERÊNCIA (o usado no treino) e gera, de forma
determinística (`--seed`), um "Dataset de Produção" que simula a passagem do
tempo deslocando intencionalmente a distribuição de features importantes:

  - `MonthlyIncome`  -> renda cresce (inflação) + variação individual;
  - `age`            -> a população envelhece;
  - `DebtRatio` e `RevolvingUtilizationOfUnsecuredLines` -> sobem;
  - contadores de atraso (30-59/60-89/90+) -> inadimplência aumenta.

É **drift de distribuição**, não corrupção: nenhum valor sai das faixas do
contrato de dados (sem nulos, negativos, sentinelas ou duplicatas). São gerados
`--stages` estágios (padrão 3) para dar a noção de passagem do tempo.

Uso:
    python scripts/simular_producao.py
    python scripts/simular_producao.py --seed 42 --stages 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credito.esquema import (
    MAX_AGE,
    MAX_PAST_DUE,
    MIN_AGE_EXCLUSIVE,
    PAST_DUE_COLUMNS,
    TARGET,
    VALID_TARGET_VALUES,
)

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_OUTPUT_DIR = RAIZ / "data" / "production"

SEED = 42
ESTAGIOS = 3
NOME_ARQUIVO = "producao_t{estagio}.csv"

# Intensidade do deslocamento por estágio temporal (cumulativo).
RENDA_POR_ESTAGIO = 0.12
RUIDO_RENDA = 0.03
IDADE_POR_ESTAGIO = 3
DIVIDA_POR_ESTAGIO = 0.15
UTILIZACAO_POR_ESTAGIO = 0.12
ATRASO_POR_ESTAGIO = 1

FEATURES_DESLOCADAS = [
    "MonthlyIncome",
    "age",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    *PAST_DUE_COLUMNS,
]
FEATURES_ESTAVEIS = [
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]


def _garantir_contrato(frame: pd.DataFrame) -> None:
    """Checagens baratas dos invariantes do contrato (falha alto se violar)."""
    if frame.duplicated().any():
        raise ValueError("Simulação gerou linhas duplicadas (contrato exige unicidade)")
    if frame["MonthlyIncome"].isna().any():
        raise ValueError("Simulação gerou renda nula (contrato exige renda não nula)")
    if (frame["age"] <= MIN_AGE_EXCLUSIVE).any() or (frame["age"] > MAX_AGE).any():
        raise ValueError("Simulação gerou idade fora da faixa do contrato")
    nao_negativas = (
        "MonthlyIncome",
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "NumberOfDependents",
    )
    for coluna in nao_negativas:
        if (frame[coluna] < 0).any():
            raise ValueError(f"Simulação gerou {coluna} negativo")
    for coluna in PAST_DUE_COLUMNS:
        if not frame[coluna].between(0, MAX_PAST_DUE).all():
            raise ValueError(
                f"Simulação gerou {coluna} fora de [0, {MAX_PAST_DUE}]"
            )
    if not frame[TARGET].isin(VALID_TARGET_VALUES).all():
        raise ValueError("Simulação alterou o target para fora de {0, 1}")


def _simular_estagio(
    reference: pd.DataFrame, estagio: int, seed: int
) -> pd.DataFrame:
    """Desloca as distribuições para o estágio `estagio` (>= 1), sem violar faixas."""
    if estagio < 1:
        raise ValueError("estagio deve ser >= 1")
    rng = np.random.default_rng(seed + estagio)
    frame = reference.copy()

    # Renda: cresce com o tempo (inflação) + variação individual contínua.
    # O ruído contínuo mantém as linhas únicas (o contrato exige sem duplicatas).
    fator_renda = 1.0 + RENDA_POR_ESTAGIO * estagio
    ruido = rng.normal(1.0, RUIDO_RENDA, size=len(frame))
    frame["MonthlyIncome"] = (frame["MonthlyIncome"] * fator_renda * ruido).clip(
        lower=0
    )

    # Idade: a população envelhece, respeitando o teto do contrato.
    frame["age"] = (frame["age"] + IDADE_POR_ESTAGIO * estagio).clip(upper=MAX_AGE)

    # Endividamento e utilização do crédito sobem ao longo do tempo.
    fator_divida = 1.0 + DIVIDA_POR_ESTAGIO * estagio
    frame["DebtRatio"] = (frame["DebtRatio"] * fator_divida).clip(lower=0)
    fator_utilizacao = 1.0 + UTILIZACAO_POR_ESTAGIO * estagio
    frame["RevolvingUtilizationOfUnsecuredLines"] = (
        frame["RevolvingUtilizationOfUnsecuredLines"] * fator_utilizacao
    ).clip(lower=0)

    # Contadores de atraso: inadimplência aumenta dentro da faixa plausível.
    for coluna in PAST_DUE_COLUMNS:
        frame[coluna] = (frame[coluna] + ATRASO_POR_ESTAGIO * estagio).clip(
            0, MAX_PAST_DUE
        )

    _garantir_contrato(frame)
    return frame


def simular_producao(
    reference: pd.DataFrame, seed: int = SEED, estagios: int = ESTAGIOS
) -> list[tuple[str, pd.DataFrame]]:
    """Gera os estágios temporais e devolve `[(nome_arquivo, frame), ...]`."""
    if estagios < 1:
        raise ValueError("estagios deve ser >= 1")
    return [
        (
            NOME_ARQUIVO.format(estagio=estagio),
            _simular_estagio(reference, estagio, seed),
        )
        for estagio in range(1, estagios + 1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera o Dataset de Produção simulando a passagem do tempo."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stages", type=int, default=ESTAGIOS)
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"[erro] {args.input} não encontrado. "
            "A referência versionada fica em data/reference/reference.csv.",
            file=sys.stderr,
        )
        return 2
    if args.stages < 1:
        print("[erro] --stages deve ser >= 1", file=sys.stderr)
        return 2

    reference = pd.read_csv(args.input)
    estagios = simular_producao(reference, seed=args.seed, estagios=args.stages)

    args.output.mkdir(parents=True, exist_ok=True)
    print("Simulação do dataset de produção concluída:")
    print(f"  - referência: {args.input} ({len(reference)} linhas)")
    print(f"  - seed: {args.seed} | estágios: {args.stages}")
    print("  - features com distribuição deslocada:")
    for nome in FEATURES_DESLOCADAS:
        print(f"      * {nome}")
    print("  - features mantidas (sem drift intencional):")
    for nome in FEATURES_ESTAVEIS:
        print(f"      * {nome}")
    for estagio, (nome, frame) in enumerate(estagios, start=1):
        destino = args.output / nome
        frame.to_csv(destino, index=False)
        print(f"  - t{estagio}: {destino} ({len(frame)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
