# Tech Challenge Fase 4 — Risco de crédito

Classificador de risco de crédito com validação de dados, simulação de drift
e monitoramento local. O objetivo é mostrar quando os dados mudam ou deixam
de atender ao contrato, usando Great Expectations, Evidently e MLflow.

Entrega: **17/11/2026**.

## Estrutura

```text
src/credito/       # preparação, contrato, modelo, drift e monitoramento
scripts/           # comandos para executar as etapas
tests/             # testes automatizados
data/reference/    # base limpa usada como entrada
reports/           # relatório HTML de drift
artifacts/         # modelos, métricas e runs gerados (fora do Git)
```

## Como executar

Use Python 3.12 e execute os comandos na raiz do projeto.
No PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,monitoramento]"
```

No Linux/macOS, ative com `source .venv/bin/activate`.

A referência limpa já está no repositório. O comando abaixo executa o fluxo
completo: preparação → contrato → treino → produção simulada → predição → drift.

```powershell
python scripts/monitorar_pipeline.py --cenario estavel
python scripts/monitorar_pipeline.py --cenario drift
python scripts/monitorar_pipeline.py --cenario invalido
```

Cada comando cria um run separado no MLflow. Os códigos de saída indicam:

| Código | Resultado |
| --- | --- |
| 0 | Cenário estável, sem drift agregado |
| 1 | Lote inválido, bloqueado antes da predição |
| 2 | Erro ou amostra insuficiente |
| 3 | Drift detectado; é um alerta, não uma falha técnica |

Para abrir a interface, execute em outro terminal com o ambiente ativo:

```powershell
python -m mlflow server --backend-store-uri sqlite:///artifacts/monitoramento/mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Acesse [MLflow local](http://127.0.0.1:5000), experimento `credito-pipeline`.
Os logs JSONL, resumo e relatórios ficam em
`artifacts/monitoramento/runs/<run_id>/`. O histórico fica no banco SQLite local.

## O que cada etapa faz

### 1. Modelo e contrato de dados

A base é [Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset):
150 mil registros, com 120.024 após a limpeza. O rótulo é `SeriousDlqin2yrs`.
O baseline usa Regressão Logística com `class_weight="balanced"`, divisão
estratificada de 80% para treino e 20% para teste, e seed 42.

O contrato do Great Expectations exige idade maior que 18 e até 120, renda
não nula, ausência de duplicatas, valores numéricos finitos e limites de domínio.
Qualquer violação reprova o lote inteiro.

- **Treino:** exige o rótulo binário e as colunas exatas do esquema.
- **Inferência:** permite rótulo ausente e colunas extras; verifica duplicatas
  pelas features. Se houver rótulo, a predição o valida antes de calcular métricas.

As regras ficam em `src/credito/contrato_dados.py` e os limites em `esquema.py`.

### 2. Simulação e drift

A produção simulada parte apenas dos 20% separados para teste (holdout).
O modelo não foi treinado nessas linhas. São gerados estágios `t1`, `t2` e `t3`,
com alterações de renda, idade, dívida, utilização de crédito e três contadores
de atraso. Parte dos rótulos também muda para simular maior inadimplência.
Os dados continuam válidos no contrato.

O Evidently aplica **KS** em cada feature: `p_value < 0,05` indica mudança.
O alerta agregado ocorre quando pelo menos **5 das 10 features** têm drift.
O **PSI** é complementar: abaixo de 0,1 é estável; de 0,1 a 0,25, moderado;
a partir de 0,25, forte. Esses limites são escolhas da demonstração.

KS/PSI comparam as entradas; não comprovam concept drift nem queda de desempenho.
O desempenho é medido separadamente quando há rótulos disponíveis.

### 3. Monitoramento

O MLflow registra status e duração das etapas, volume processado, regras
violadas, KS/PSI, alertas e métricas do modelo. Sem rótulos, as métricas de
desempenho ficam indisponíveis. CPU, RAM e disco são medições pontuais da máquina.

Os cenários são testes separados: `estavel` usa o holdout sem alterações;
`drift` simula a crise; `invalido` coloca uma idade de 16 anos para demonstrar
o bloqueio. Não há servidor contínuo de inferência nem retreino automático.

## Resultados da demonstração

Execução com a base real e seed 42:

| Cenário | ROC AUC | Features com drift | Resultado |
| --- | --- | --- | --- |
| Estável | 0,8018 | 0/10 | Sem alerta |
| Crise simulada (`t3`) | 0,6623 | 7/10 | Alerta de drift |
| Inválido | Não calculada | Não calculado | Contrato bloqueia o lote |

A AUC mede a capacidade de ordenar clientes pelo risco, não a porcentagem de
acertos. A queda na simulação mostra por que o monitoramento é necessário.
O baseline tem precision 0,2501 e recall 0,5949; o desbalanceamento da base
(6,89% de inadimplentes) e o peso balanceado afetam esse resultado.

## Comandos por etapa

Para executar as etapas individualmente:

```powershell
python scripts/treinar_modelo.py
python scripts/gerar_lote_corrompido.py
python scripts/validar_lote.py data/corrupted/corrupted_batch.csv
python scripts/simular_producao.py
python scripts/predizer.py --input data/production/producao_t3.csv --output data/production/predicoes_t3.csv
python scripts/detectar_drift.py data/production/producao_t3.csv
```

Use `--perfil inferencia` no validador para lotes sem rótulo ou com predições.
Cada comando aceita `--help`. O modelo e os JSONs ficam em `artifacts/`;
o HTML fica em `reports/drift_report.html`. As saídas padrão são sobrescritas.

No comando isolado de drift, a referência padrão inclui o holdout. Para uma
comparação sem essa sobreposição, use uma referência só de treino com
`--reference`, ou prefira o pipeline monitorado, que já faz essa separação.

Para refazer a base a partir do Kaggle:

```powershell
python scripts/baixar_dataset.py
python scripts/preparar_referencia.py
```

Se o download falhar, o projeto gera dados sintéticos e avisa no console.
Para gerar somente dados sintéticos, use `baixar_dataset.py --force-synthetic`;
acrescente `--force` para substituir um arquivo bruto existente. A preparação
sobrescreve a referência. Carregue arquivos `.joblib` apenas de fontes confiáveis.

## Verificação

```powershell
ruff check .
pytest
```

O GitHub Actions executa lint, testes e o pipeline, incluindo os três cenários,
e publica os relatórios e logs como artefatos. O HTML versionado é um exemplo
de execução anterior; para evidências atuais, consulte os runs ou gere novamente.

## Etapa 4 — Governança, LGPD e causalidade

Projeto acadêmico com dados públicos do Give Me Some Credit, sem concessão
real de crédito. Esta etapa é documental.

### Dados e privacidade

O dataset não contém nome ou CPF, mas idade, renda e histórico financeiro
podem contribuir para identificar pessoas. Não afirmamos anonimização
comprovada; dados públicos também exigem cuidado.

A limpeza remove o índice original, nulos e duplicatas. Os logs padrão
registram métricas e nomes de regras, sem linhas de clientes. Dados gerados,
modelos e banco MLflow ficam fora do Git. Antes de compartilhar relatórios,
revisar seu conteúdo e não cruzar a base com fontes para reidentificação.
Para demonstrações sem registros reais, usar `baixar_dataset.py --force-synthetic`.

### Base legal e responsabilidades

Em um uso real, a fintech seria a controladora; fornecedores que tratassem
os dados em seu nome seriam operadores. A hipótese de proteção do crédito
(art. 7º, X, da LGPD) precisaria ser validada para a finalidade do tratamento.
Usar uma base pública não dispensa essa análise. O mantenedor é responsável
pelos arquivos e pela revisão das publicações deste projeto.

### Retenção e direitos

Política proposta: guardar os arquivos até a avaliação final + 90 dias e,
depois, excluir manualmente dados e derivados locais, modelos, runs e caches.
Revisar também dados no Git, histórico, backups e artefatos da CI; o projeto
não apaga automaticamente cópias ou forks. Os 90 dias são uma escolha do
projeto, não um prazo imposto pela LGPD.

Antes de uso real, oferecer acesso e correção dos dados e revisão de decisões
automatizadas (arts. 18 e 20), com análise humana como política do serviço.
Esses canais não estão implementados neste protótipo.

### Vieses, causalidade e resposta

Excluir registros sem renda e usar idade/renda pode afetar grupos de forma
diferente. A revisão proposta é comparar precision, recall e erros por faixa
de idade e renda; essa auditoria ainda não foi executada.

A hipótese da simulação é: crise econômica → mudança em renda, dívida e
atrasos → possível queda de desempenho. Os deslocamentos e as mudanças de
rótulo são artificiais e não provam causalidade no mundo real. KS/PSI medem
mudança nas entradas; a queda de desempenho é verificada pelas métricas com
rótulo, comparadas ao cenário estável.

O contrato já bloqueia lotes inválidos. Em caso de falha ou drift, consultar
o run e os relatórios; investigar os dados antes de decidir por retreino.
Não há retreino automático. Retenção, atendimento ao titular e auditoria de
viés são propostas, sem automação nesta entrega.

Referência: [LGPD — Lei nº 13.709/2018](https://www2.camara.leg.br/legin/fed/lei/2018/lei-13709-14-agosto-2018-787077-normaatualizada-pl.html).
