"""Etapa 2 (1/3) — Dataset de Produção simulado (passagem do tempo).

A produção parte do **holdout** da referência (os 20% que o modelo não viu no
treino, ver `credito.modelo.dividir_referencia`) e gera, de forma determinística
(`--seed`), estágios t1, t2, t3... que simulam uma piora da economia:

Data drift (muda a distribuição de X):
  - `MonthlyIncome` -> renda nominal sobe para todo mundo (inflação);
  - `age`           -> a carteira envelhece;
  - `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` e contadores de atraso
    -> sobem, mas só para uma fração dos clientes (os "atingidos pela crise"),
    que cresce a cada estágio (20%, 40%, 60%...).

Concept drift (muda a relação entre X e y):
  - entre os clientes que já eram bem endividados, parte passa a ficar
    inadimplente de verdade (`SeriousDlqin2yrs` 0 -> 1). Para o mesmo perfil
    de X, o risco real ficou maior do que o modelo aprendeu.

Nada disso é corrupção: os valores continuam dentro das faixas do contrato.

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
from credito.modelo import dividir_referencia

RAIZ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RAIZ / "data" / "reference" / "reference.csv"
DEFAULT_OUTPUT_DIR = RAIZ / "data" / "production"

SEED = 42
ESTAGIOS = 3
NOME_ARQUIVO = "producao_t{estagio}.csv"

# Inflação e envelhecimento: valem para a carteira toda, acumulando por estágio.
RENDA_POR_ESTAGIO = 0.12
RUIDO_RENDA = 0.03
IDADE_POR_ESTAGIO = 3

# Crise: a cada estágio, mais 20% dos clientes são "atingidos". Quem é atingido
# leva sempre o mesmo choque; o que cresce com o tempo é quantos são atingidos.
# (Na primeira versão eu somava atraso em todo mundo, o que não é realista.)
FRACAO_ATINGIDA_POR_ESTAGIO = 0.20
CHOQUE_DIVIDA = 1.5
CHOQUE_UTILIZACAO = 1.3
CHOQUE_ATRASO = 1

# Concept drift: entre os clientes do quartil mais endividado, uma parte dos
# adimplentes passa a ser inadimplente (5%, 10%, 15%...).
QUANTIL_ENDIVIDADO = 0.75
VIRADA_ROTULO_POR_ESTAGIO = 0.05

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

    # Quem era endividado ANTES da crise (usado no concept drift lá embaixo).
    limite_endividado = frame["DebtRatio"].quantile(QUANTIL_ENDIVIDADO)
    endividado = (frame["DebtRatio"] >= limite_endividado).to_numpy()

    # Crise: sorteia os clientes atingidos neste estágio.
    fracao = min(FRACAO_ATINGIDA_POR_ESTAGIO * estagio, 1.0)
    atingido = rng.random(len(frame)) < fracao

    frame.loc[atingido, "DebtRatio"] *= CHOQUE_DIVIDA
    frame.loc[atingido, "RevolvingUtilizationOfUnsecuredLines"] *= CHOQUE_UTILIZACAO
    for coluna in PAST_DUE_COLUMNS:
        frame.loc[atingido, coluna] = (
            frame.loc[atingido, coluna] + CHOQUE_ATRASO
        ).clip(0, MAX_PAST_DUE)

    # Concept drift: com juros altos, parte de quem já era endividado e pagava
    # em dia passa a não conseguir pagar. As features dessas pessoas não mudam
    # (fora inflação/idade), mas o rótulo muda -> P(y | X) mudou.
    adimplente = (frame[TARGET] == 0).to_numpy()
    chance_virar = min(VIRADA_ROTULO_POR_ESTAGIO * estagio, 1.0)
    sorteado = rng.random(len(frame)) < chance_virar
    vira_inadimplente = endividado & adimplente & sorteado
    frame.loc[vira_inadimplente, TARGET] = 1

    _garantir_contrato(frame)
    return frame


def simular_producao(
    reference: pd.DataFrame, seed: int = SEED, estagios: int = ESTAGIOS
) -> list[tuple[str, pd.DataFrame]]:
    """Gera os estágios temporais a partir de `reference` (no CLI, o holdout)."""
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
    # Só o holdout vira produção: são clientes que o modelo nunca viu no treino.
    _, holdout = dividir_referencia(reference)
    holdout = holdout.reset_index(drop=True)
    estagios = simular_producao(holdout, seed=args.seed, estagios=args.stages)

    args.output.mkdir(parents=True, exist_ok=True)
    print("Simulação do dataset de produção concluída:")
    print(f"  - referência: {args.input} ({len(reference)} linhas)")
    print(f"  - base da produção: holdout ({len(holdout)} linhas, fora do treino)")
    print(f"  - seed: {args.seed} | estágios: {args.stages}")
    print("  - features com distribuição deslocada:")
    for nome in FEATURES_DESLOCADAS:
        print(f"      * {nome}")
    print("  - features mantidas (sem drift intencional):")
    for nome in FEATURES_ESTAVEIS:
        print(f"      * {nome}")
    taxa_base = holdout[TARGET].mean()
    for estagio, (nome, frame) in enumerate(estagios, start=1):
        destino = args.output / nome
        frame.to_csv(destino, index=False)
        print(
            f"  - t{estagio}: {destino} ({len(frame)} linhas, "
            f"inadimplência real {taxa_base:.2%} -> {frame[TARGET].mean():.2%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
