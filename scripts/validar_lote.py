"""Valida um CSV contra o contrato de dados (exit 1 = ingestão bloqueada).

Uso:
    python scripts/validar_lote.py data/reference/reference.csv
    python scripts/validar_lote.py data/corrupted/corrupted_batch.csv
"""

from credito.validacao_lote import main

if __name__ == "__main__":
    raise SystemExit(main())
