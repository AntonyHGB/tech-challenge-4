"""Baixa o dataset Give Me Some Credit (fallback sintético se o Kaggle falhar).

Uso:
    python scripts/baixar_dataset.py
    python scripts/baixar_dataset.py --force-synthetic
"""

from credito.aquisicao import main

if __name__ == "__main__":
    raise SystemExit(main())
