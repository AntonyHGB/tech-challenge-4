"""Gera o Dataset de Produção simulando a passagem do tempo (drift de distribuição).

Uso:
    python scripts/simular_producao.py
    python scripts/simular_producao.py --seed 42 --stages 3
"""

from credito.simulacao_producao import main

if __name__ == "__main__":
    raise SystemExit(main())
