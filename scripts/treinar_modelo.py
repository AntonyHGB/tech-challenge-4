"""Treina o baseline de risco de crédito e grava os artefatos em artifacts/.

Uso:
    python scripts/treinar_modelo.py
    python scripts/treinar_modelo.py --input data/reference/reference.csv
"""

from credito.modelo import main

if __name__ == "__main__":
    raise SystemExit(main())
