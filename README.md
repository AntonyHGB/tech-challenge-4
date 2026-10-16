# Tech Challenge Fase 4 — Etapas 1, 2 e 3: Contratos, Drift e Monitoramento

MVP das Etapas 1, 2 e 3: um classificador binário baseline de **risco de crédito**
(Give Me Some Credit) protegido por um **contrato de dados** com Great
Expectations, que **bloqueia a ingestão** de lotes fora do contrato, mais a
demonstração de **drift** sobre um Dataset de Produção simulado, e o
monitoramento local do pipeline com **MLflow** (runs, logs JSONL e métricas
de recursos) na Etapa 3.

## O que este MVP entrega

1. **Classificador baseline** treinado somente com o dataset de **Referência** (limpo):
   Regressão Logística, split estratificado e fixo 80/20 (`dividir_referencia`,
   seed 42), AUC/precision/recall/accuracy.
2. **Contrato de dados** com Great Expectations: regras rígidas (sem `mostly`),
   em dois perfis — `treino` (exige o rótulo) e `inferencia` (lote de produção:
   sem rótulo e com colunas extras aceitas).
3. **Demonstração de bloqueio**: um lote corrompido é reprovado pelo contrato e a
   ingestão é barrada com **exit code 1**.
4. **Etapa 2**: **Dataset de Produção** simulado a partir do **holdout** da
   referência (passagem do tempo, 7 features com distribuição deslocada e
   viradas de rótulo sintéticas), **predições** do baseline e **relatório de
   drift** com Evidently — HTML em `reports/` + JSON — distinguindo drift de
   violação do contrato (KS decide; PSI é leitura complementar).
5. **Etapa 3**: monitoramento **local** do pipeline com **MLflow** — o novo
   CLI registra cada execução como *run* no experimento `credito-pipeline`,
   com logs JSONL, métricas de contrato, drift, performance e recursos, e
   persistência em SQLite; a UI é apenas local e manual.

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
│   ├── modelo.py               # treina o baseline -> artifacts/
│   ├── drift.py                # drift: relatório JSON + exit codes (+ HTML)
│   ├── relatorio_drift.py      # relatório HTML do Evidently -> reports/
│   ├── simulacao_producao.py   # Dataset de Produção (a partir do holdout)
│   └── predicao.py             # predições do baseline -> CSV
├── scripts/                # CLIs finos sobre o pacote
│   ├── baixar_dataset.py
│   ├── preparar_referencia.py
│   ├── validar_lote.py
│   ├── gerar_lote_corrompido.py
│   ├── treinar_modelo.py
│   ├── simular_producao.py
│   ├── detectar_drift.py
│   ├── predizer.py
│   └── monitorar_pipeline.py
├── tests/                  # pytest (lotes sintéticos, sem rede/Kaggle)
├── data/
│   ├── raw/            # CSV original do Kaggle (não versionado)
│   ├── reference/      # dataset de referência limpo (versionado)
│   ├── production/     # Dataset de Produção simulado (não versionado)
│   └── corrupted/      # lote corrompido gerado (não versionado)
├── artifacts/          # model.joblib + baseline_metrics.json + drift_report.json;
│                       #   monitoramento/ (runs e mlflow.db do MLflow; não versionado)
├── reports/            # relatório de drift HTML do Evidently (versionado)
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Como reproduzir

Pré-requisito: Python 3.12. O projeto é um pacote instalável (`pip install -e ".[dev]"`)
e os comandos abaixo rodam da raiz do repositório. No Windows, crie o venv
com `python -m venv .venv`. Ative com `source .venv/Scripts/activate` no Git Bash,
`.venv\Scripts\activate.bat` no CMD ou `.venv\Scripts\Activate.ps1` no PowerShell.
Sem ativar, use `.venv/Scripts/python.exe scripts/...` no Git Bash.

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

# --- Etapa 2: Dataset de Produção, predições e drift ---

# 7) Simular o Dataset de Produção (a partir do holdout -> data/production/)
python scripts/simular_producao.py

# 8) Gerar as predições do baseline para o Dataset de Produção
python scripts/predizer.py --input data/production/producao_t3.csv \
  --output data/production/predicoes_t3.csv

# 8b) Validar as predições no perfil inferencia (aceita colunas extras
#     e rótulo ausente — um CSV sem SeriousDlqin2yrs também passa)
python scripts/validar_lote.py data/production/predicoes_t3.csv --perfil inferencia

# 9) Detectar drift -> relatório JSON + HTML em reports/
python scripts/detectar_drift.py data/production/producao_t3.csv
echo $?   # 3 (drift detectado); HTML em reports/drift_report.html
```

Cada script é um CLI fino sobre o módulo correspondente do pacote `credito`,
que também pode ser executado via módulo (ex.: `python -m credito.validacao_lote`).
O CI (`.github/workflows/ci.yml`) roda `ruff check .`, `pytest` e o job
`pipeline` (treino -> produção -> predições -> validação do perfil
`inferencia` -> drift -> monitoramento nos três cenários), publicando os
relatórios HTML e JSON e os logs do monitoramento como *artifact*.

Os passos 2 e 3 **sobrescrevem** seus arquivos de saída
(`data/reference/reference.csv` e `artifacts/`) a cada execução; só o passo 1
(`baixar_dataset.py`) tem guarda contra sobrescrita — use `--force` nele se
precisar baixar de novo.

## Contrato de dados (regras rígidas, sem tolerância `mostly`)

Qualquer linha violando qualquer regra reprova o lote inteiro (sem tolerância `mostly`).

O contrato opera em **dois perfis** (`construir_regras()` em
`src/credito/contrato_dados.py` é a fonte definitiva): nesta revisão são
**33 regras no perfil `treino`** e **31 no perfil `inferencia`**, mas o número
pode evoluir junto com o código — confira lá antes de citar contagens.

- **`treino`** (padrão; vale para a referência): exige o rótulo
  `SeriousDlqin2yrs` e exatamente as colunas do esquema.
- **`inferencia`** (lotes de produção, ex.: CSV de predições): dispensa o
  rótulo (em crédito ele só chega meses depois), aceita colunas extras e checa
  duplicatas sobre as features.
  Esse perfil valida as features: se o target estiver presente, seu conteúdo
  não é avaliado pelo contrato de inferência. A predição valida separadamente
  o rótulo presente antes de calcular métricas; para um dataset de treino,
  use o perfil `treino`.

Além das regras da tabela abaixo, o contrato foi endurecido com **guardas de
sanidade por feature**: cada coluna numérica precisa ser **não nula e
finita/numérica** (protege o treino e o cálculo de drift contra NaN/infinitos).
A tabela a seguir é o **inventário legado das regras originais da Etapa 1**,
mantida como referência:

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

"Teste" aqui é o **holdout** de `dividir_referencia` (80/20 estratificado, fixo,
seed 42): 96.019 linhas de treino e 24.005 de holdout sobre os 120.024 da
referência. Os artefatos gerados (`artifacts/model.joblib`,
`artifacts/baseline_metrics.json`) **não são versionados**. Regenere o modelo
com `python scripts/treinar_modelo.py` neste checkout e carregue arquivos
`.joblib` somente de fontes confiáveis: a desserialização pode executar código.
A produção simulada deriva dos dados de referência, não do modelo; somente as
predições dependem do modelo treinado.

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

Agregação/alertas contínuos de observabilidade (Prometheus/Grafana),
documentação LGPD (Etapa 4) e vídeo.

## Etapa 2 — Dataset de Produção, predições e drift

O fluxo completo da etapa é (com o pacote instalado em `pip install -e ".[dev]"`):

```bash
python scripts/simular_producao.py       # data/production/producao_t{1,2,3}.csv (do holdout)
python scripts/predizer.py --input data/production/producao_t3.csv \
  --output data/production/predicoes_t3.csv
python scripts/validar_lote.py data/production/predicoes_t3.csv --perfil inferencia
python scripts/detectar_drift.py data/production/producao_t3.csv
echo $?   # 3 (drift detectado) -> artifacts/drift_report.json + reports/drift_report.html
```

O **Dataset de Produção** (`simulacao_producao.py`) parte do **holdout** da
Referência — os 20% que o modelo não viu no treino, cortados pelo mesmo
`dividir_referencia` (estratificado e fixo) usado no treino — e desloca
intencionalmente a distribuição de 7 features para simular a passagem do tempo:
`MonthlyIncome` (renda cresce com a inflação + ruído individual), `age` (a
população envelhece), `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` e os 3
contadores de atraso. A cada estágio a "crise" atinge uma fração maior de
clientes (~20%, 40%, 60% em t1..t3) com choque de dívida ×1,5, de utilização
×1,3 e +1 atraso nos contadores. A geração é **determinística** pela `--seed` e
produz `--stages` estágios (padrão 3, `t1..t3`) em `data/production/`.

Há ainda **concept drift simulado**: entre os clientes do quartil mais
endividado, uma fração crescente de adimplentes tem o rótulo sinteticamente
virado para 1 (0 -> 1), além dos deslocamentos das features — a relação com o
rótulo é alterada de propósito. Isso
é **simulação**, não evidência medida: o detector de drift compara apenas as
**features** (KS/PSI), então não "detecta" essa mudança de conceito, e a
simulação **não garante** degradação de métricas em um estágio específico (as
métricas por estágio só aparecem no console quando o rótulo está presente).
Nada disso é corrupção: os valores permanecem dentro das faixas do contrato e o
lote **passa** nele.

O `predizer.py` carrega `artifacts/model.joblib` (falha com mensagem clara se o
modelo não existir) e grava um CSV com `probabilidade_inadimplencia` e
`classe_prevista`, preservando as colunas de entrada (inclusive extras); se o
rótulo estiver presente, reporta métricas no console, caso contrário avisa que
não há métricas. A predição valida as features no contrato de inferência antes
de carregar o modelo; rótulos presentes devem ser binários e não nulos. Se há
uma única classe no lote, a ROC AUC é `N/A`, sem impedir as demais métricas.
A saída não pode coincidir com a entrada ou com o modelo.
O CSV de predições é lote de produção: valide-o com
`--perfil inferencia` (aceita rótulo ausente e colunas extras).

O `detectar_drift.py` mantém o **JSON** (`artifacts/drift_report.json`) e os
**exit codes** e ainda grava o **relatório HTML do Evidently** (`DataDriftPreset`)
em `reports/drift_report.html` (**versionado** no repositório), apontando as
variáveis com drift. `--html` troca esse destino. O **PSI** entra como leitura
complementar no JSON e no console (níveis estável/moderado/forte); quem decide o
exit code é o **KS**, e o HTML mostra o mesmo critério do JSON. O próprio drift
valida o lote no perfil `inferencia` (produção sem rótulo ou já com predições).
O CI apaga o HTML do checkout antes do pipeline (para nunca publicar relatório
obsoleto) e publica o HTML e o JSON recém-gerados como *artifact*.

### Comparação pontual (Referência × lote)

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
(5/10) têm drift. Ambos os limites são ajustáveis no CLI. O **PSI** é calculado
por feature como leitura complementar (faixas de bolso: < 0,1 estável; 0,1–0,25
moderado; >= 0,25 forte): aparece no JSON e no console, mas **não** decide o
exit code nem entra no HTML. O mínimo padrão é
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

⚠️ Limitações e avisos importantes:

- **A referência default inclui o holdout.** O `--reference` padrão é
  `data/reference/reference.csv`, o conjunto **completo** (120.024 linhas), que
  **inclui o holdout** de onde a produção simulada foi gerada. Como referência
  e lote se sobrepõem, essa comparação serve para **monitoramento e
  demonstração do fluxo**, não para avaliação estatística independente do
  detector. Para comparar com separação real, passe uma referência própria via
  `--reference` (ex.: um CSV sem o holdout) — o design atual não faz isso
  automaticamente.
- **Rótulo:** o perfil `inferencia` aceita lotes de produção **sem**
  `SeriousDlqin2yrs` (o próprio `detectar_drift.py` valida o lote nesse perfil);
  a **referência**, porém, continua no perfil `treino` e precisa do rótulo.
- O relatório JSON padrão em `artifacts/` não é versionado e é sobrescrito a cada
  execução; use `--report` para preservar execuções. O HTML fica em `reports/`
  (versionado) e também é sobrescrito a cada execução; use `--html` para trocar o
  destino. Se uma execução falhar, o relatório anterior é removido antes da
  leitura; confira o exit code `2`, pois pode não existir JSON nessa situação.
- `--report` e `--html` não podem apontar para um CSV de entrada nem para o
  mesmo arquivo entre si. A API de PSI rejeita entradas vazias, não numéricas
  ou não finitas; a referência constante usa faixas abaixo/igual/acima do valor.
  Drift não
  equivale a degradação de performance nem a violação do contrato. O lote
  corrompido é para testar o contrato, **não** para demonstrar drift. Não há
  agendamento nem alertas contínuos nesta etapa.

## Etapa 3 — Monitoramento do pipeline (MLflow local)

Monitoramento **local e simples** do pipeline com **MLflow**: o CLI
`scripts/monitorar_pipeline.py` reusa os módulos existentes (preparação,
contrato, treino com o mesmo split estratificado 80/20 do baseline,
simulação de produção, validação do perfil `inferencia`, predição e drift)
e grava cada execução como *run* no experimento `credito-pipeline`. A
referência do monitoramento é **só o treino** (o holdout fica separado);
no cenário estável, o holdout entra como t0, sem deslocamento.

```bash
pip install -e ".[dev,monitoramento]"

# Estável (holdout t0, sem deslocamento) -> exit 0
python scripts/monitorar_pipeline.py --cenario estavel
echo $?   # 0

# Drift (produção simulada) -> exit 3; drift é RESULTADO, não erro
python scripts/monitorar_pipeline.py --cenario drift
echo $?   # 3

# Inválido -> exit 1; o contrato BLOQUEIA antes de predizer
python scripts/monitorar_pipeline.py --cenario invalido
echo $?   # 1
```

Os defaults são `--input data/reference/reference.csv`,
`--output artifacts/monitoramento` e `--estagio 3`. Códigos de saída: `0`
estável, `1` bloqueado por invalidade, `2` erro (incluindo dados
insuficientes) e `3` drift detectado. Repita os três cenários no mesmo
`--output` para ver o histórico na UI.

Cada run registra: regras do contrato, volume de linhas e duração das
etapas; KS/PSI por feature (o **KS** decide; o **PSI** é leitura
complementar — degradação de performance **não** é comprovada por PSI);
métricas de performance **apenas quando há rótulo** (lote sem rótulo não
gera métricas e não produz falso 0); e uso de recursos por **snapshot** da
execução — CPU %, RAM % e disco % do host e RSS do processo em bytes. Não
há agente nem daemon.

Para inspecionar a UI **manualmente** (nunca no CI), rode e abra
`http://127.0.0.1:5000`:

```bash
python -m mlflow server \
  --backend-store-uri sqlite:///artifacts/monitoramento/mlflow.db \
  --host 127.0.0.1 --port 5000 --workers 1
```

O run ID aparece no console ao iniciar cada execução; os logs JSONL e os
artefatos (resumo, qualidade, relatório de drift JSON/HTML) ficam
centralizados em `artifacts/monitoramento/runs/<run_id>/`. A persistência é
**SQLite local** (`artifacts/monitoramento/mlflow.db`); os diretórios
gerados já estão ignorados no Git. **Não** versione o banco e **não**
registre PII nem CSVs com dados de clientes.

⚠️ A referência do monitoramento **novo** é só o treino (holdout separado).
A ressalva do CLI legado permanece: o `detectar_drift.py` da Etapa 2
compara por padrão contra a Referência **completa**, que inclui o holdout —
veja as limitações na seção anterior.
