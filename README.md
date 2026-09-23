# Tech Challenge Fase 4 — Etapas 1 e 2: Contratos e Drift de Dados

MVP da Etapa 1: um classificador binário baseline de **risco de crédito**
(Give Me Some Credit) protegido por um **contrato de dados** com Great
Expectations, que **bloqueia a ingestão** de lotes fora do contrato.

## O que este MVP entrega

1. **Classificador baseline** treinado somente com o dataset de **Referência** (limpo):
   Regressão Logística, split estratificado 80/20, AUC/precision/recall/accuracy.
2. **Contrato de dados** com Great Expectations: 13 regras rígidas (sem `mostly`).
3. **Demonstração de bloqueio**: um lote corrompido é reprovado pelo contrato e a
   ingestão é barrada com **exit code 1**.
4. **Etapa 2**: comparação pontual de features com Evidently, após validar o
   contrato; gera relatório JSON e distingue drift de violação do contrato.

## Estrutura

```
tech-challenge-4/
├── src/credito/            # pacote instalável (pip install -e ".[dev]")
│   ├── esquema.py              # fonte única de verdade: colunas e limites do domínio
│   ├── aquisicao.py            # baixa o dataset do Kaggle (fallback sintético)
│   ├── preparacao.py           # limpeza -> data/reference/reference.csv
│   ├── contrato_dados.py       # suite de expectations do Great Expectations
│   ├── validacao_lote.py       # valida um lote (exit 1 = ingestão bloqueada)
│   ├── lote_corrompido.py      # gera o lote corrompido de demonstração
│   └── modelo.py               # treina o baseline -> artifacts/
│   └── drift.py                # comparação pontual de features -> relatório JSON
├── scripts/                # CLIs finos sobre o pacote
│   ├── baixar_dataset.py
│   ├── preparar_referencia.py
│   ├── validar_lote.py
│   ├── gerar_lote_corrompido.py
│   └── treinar_modelo.py
│   └── detectar_drift.py
├── tests/                  # pytest (lotes sintéticos, sem rede/Kaggle)
├── data/
│   ├── raw/            # CSV original do Kaggle (não versionado)
│   ├── reference/      # dataset de referência limpo (versionado)
│   └── corrupted/      # lote corrompido gerado (não versionado)
├── artifacts/          # model.joblib + baseline_metrics.json (não versionado)
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Como reproduzir

Pré-requisito: Python 3.12. O projeto é um pacote instalável (`pip install -e ".[dev]"`)
e os comandos abaixo rodam da raiz do repositório.

```bash
cd tech-challenge-4
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1) Baixar o dataset de referência (Give Me Some Credit)
python scripts/baixar_dataset.py

# 2) Construir o dataset de REFERÊNCIA (limpo) usado no treino
python scripts/preparar_referencia.py

# 3) Treinar o baseline
python scripts/treinar_modelo.py

# 4) Validar a referência -> PASS, exit code 0
python scripts/validar_lote.py data/reference/reference.csv
echo $?   # 0

# 5) Gerar o lote corrompido de demonstração
python scripts/gerar_lote_corrompido.py

# 6) Validar o lote corrompido -> FAIL, exit code 1 (ingestão BLOQUEADA)
python scripts/validar_lote.py data/corrupted/corrupted_batch.csv
echo $?   # 1
```

Cada script é um CLI fino sobre o módulo correspondente do pacote `credito`,
que também pode ser executado via módulo (ex.: `python -m credito.validacao_lote`).
O CI (`.github/workflows/ci.yml`) roda `ruff check .` e `pytest`.

Os passos 2 e 3 **sobrescrevem** seus arquivos de saída
(`data/reference/reference.csv` e `artifacts/`) a cada execução; só o passo 1
(`baixar_dataset.py`) tem guarda contra sobrescrita — use `--force` nele se
precisar baixar de novo.

## Contrato de dados (13 regras rígidas)

Qualquer linha violando qualquer regra reprova o lote inteiro (sem tolerância `mostly`).

| Regra | Expectation | O que barra |
| --- | --- | --- |
| `colunas_esperadas` | `ExpectTableColumnsToMatchSet` | esquema divergente |
| `lote_nao_vazio` | `ExpectTableRowCountToBeBetween` | lote vazio |
| `idade_maior_que_18` | `ExpectColumnValuesToBeBetween(strict_min)` | `age <= 18` |
| `renda_nao_nula` | `ExpectColumnValuesToNotBeNull` | `MonthlyIncome` nula |
| `sem_duplicatas` | `ExpectCompoundColumnsToBeUnique` | linhas duplicadas |
| `target_binario` | `ExpectColumnValuesToBeInSet` | target fora de `{0, 1}` |
| `MonthlyIncome_nao_negativo` | `ExpectColumnValuesToBeBetween(min_value=0)` | renda negativa |
| `RevolvingUtilizationOfUnsecuredLines_nao_negativo` | idem | utilização negativa |
| `DebtRatio_nao_negativo` | idem | índice de endividamento negativo |
| `NumberOfDependents_nao_negativo` | idem | dependentes negativos |
| `NumberOfTime30-59DaysPastDueNotWorse_em_faixa_plausivel` | idem (`0..50`) | sentinelas 96/98 |
| `NumberOfTimes90DaysLate_em_faixa_plausivel` | idem (`0..50`) | sentinelas 96/98 |
| `NumberOfTime60-89DaysPastDueNotWorse_em_faixa_plausivel` | idem (`0..50`) | sentinelas 96/98 |

## Referência vs. lote corrompido

O dataset original do Kaggle tem 150.000 linhas com problemas reais:
29.731 nulos em `MonthlyIncome`, 3.924 em `NumberOfDependents`, 609 linhas
duplicadas quando se ignora a coluna de índice (depois do `dropna` restam 99
duplicatas, que são as removidas na limpeza), 1 registro com idade 0 e 145
linhas com os códigos de sentinela 96/98 nos contadores de atraso. A limpeza
produz a referência com **120.024 linhas** e taxa de inadimplência de
**6,89%**.

O lote corrompido parte dessa referência e reintroduz os defeitos de
propósito (idade <= 18, renda nula, negativos, sentinelas, target inválido e
linhas duplicadas) para forçar o contrato a falhar.

## Baseline (evidência real de execução)

| Métrica | Valor |
| --- | --- |
| ROC AUC | 0.8018 |
| Accuracy | 0.8492 |
| Precision | 0.2501 |
| Recall | 0.5949 |
| F1 | 0.3522 |

Matriz de confusão no teste (24.005 amostras): `[[19401, 2950], [670, 984]]`.
O desbalanceamento (6,89% de positivos) explica a precision baixa com
`class_weight='balanced'`, que prioriza recall na classe minoritária.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
(150k linhas, target `SeriousDlqin2yrs`), baixado via `kagglehub`. O download
público funciona sem autenticação.

Se o Kaggle exigir credenciais no ambiente de execução, `baixar_dataset.py`
gera automaticamente um dataset **sintético** de fallback (6.000 amostras,
mesmas colunas) e avisa no console — o MVP roda de ponta a ponta de qualquer
forma. Para usar o dado real depois, autentique com `kagglehub.login()` ou
coloque `~/.kaggle/kaggle.json` e rode `python scripts/baixar_dataset.py --force`.

## Fora de escopo (outras etapas)

Monitoramento contínuo/observabilidade (Prometheus/Grafana, Etapa 3),
documentação LGPD (Etapa 4) e vídeo.

## Etapa 2 — detecção pontual de drift

Com o pacote instalado (`pip install -e ".[dev]"`) e a referência pronta:

```bash
python scripts/detectar_drift.py data/reference/reference.csv
# RESULTADO: sem_drift (exit code 0); relatório: .../artifacts/drift_report.json

python scripts/detectar_drift.py data/corrupted/corrupted_batch.csv
# RESULTADO: lote_invalido (exit code 1); relatório: .../artifacts/drift_report.json

# Para comparar outro lote válido sem sobrescrever o relatório padrão:
python scripts/detectar_drift.py caminho/lote.csv \
  --reference data/reference/reference.csv --report caminho/relatorio.json \
  --min-rows 100 --p-value 0.05 --drift-share 0.5
```

**Decisões iniciais, revisáveis:** Evidently **0.7.23**; método **Kolmogorov–Smirnov
(KS)** para as 10 features numéricas (`FEATURES`), com drift por coluna quando
`p_value < 0.05`. Há drift agregado quando **pelo menos 50%** das features
(5/10) têm drift. Ambos os limites são ajustáveis no CLI. O mínimo padrão é
**100 linhas em cada dataset** (`--min-rows`); abaixo dele não se calcula drift.
Os limites são escolhas operacionais para demonstração, não calibração de risco
nem garantia estatística (há múltiplas comparações). O target nunca entra no
cálculo de drift.

O contrato existente é aplicado **primeiro à referência e ao lote**. Códigos
de saída: `0` sem drift, `1` lote fora do contrato, `2` erro (incluindo amostra
insuficiente e referência inválida) e `3` drift detectado. O JSON registra
`status`, caminhos relativos ao diretório de trabalho (ou só o nome para
arquivos externos a ele) e SHA-256 de ambos os CSVs, número de linhas,
configuração, `p_value` e `drift` por feature, e contagem/share agregados. Para
`lote_invalido`, registra regras violadas; para `amostra_insuficiente`, não há
resultado estatístico. Por exemplo, com o lote de referência como lote atual:

```json
{"status": "sem_drift", "agregado": {"colunas_com_drift": 0,
 "total_colunas": 10, "share": 0.0, "drift": false}}
```

⚠️ Limitações: o contrato da Etapa 1 exige `SeriousDlqin2yrs` mesmo que o
teste de drift ignore o target; lotes ainda sem rótulo não passam no contrato
sem uma mudança futura nesse fluxo. O relatório padrão em `artifacts/` não é
versionado e é sobrescrito a cada execução; use `--report` para preservar
execuções. Se uma execução falhar, o relatório anterior é removido antes da
leitura; confira o exit code `2`, pois pode não existir JSON nessa situação.
`--report` não pode apontar para um CSV de entrada. Drift não equivale a
degradação de performance nem a violação do contrato. O lote corrompido é
para testar o contrato, **não** para demonstrar
drift. Não há agendamento nem alertas contínuos nesta etapa.
