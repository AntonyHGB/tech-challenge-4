"""Gera predições do baseline (probabilidade + classe) para um lote/dataset.

Uso:
    python scripts/predizer.py
    python scripts/predizer.py --input data/production/producao_t3.csv \
        --output data/production/predicoes_t3.csv
"""

from credito.predicao import main

if __name__ == "__main__":
    raise SystemExit(main())
