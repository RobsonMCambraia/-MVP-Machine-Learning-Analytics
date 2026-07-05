# MVP — Predição de Doença Cardíaca com Machine Learning

Projeto de Machine Learning para prever a presença ou ausência de doença cardíaca em pacientes a partir de indicadores clínicos, desenvolvido como MVP (Minimum Viable Product) de pós-graduação em Ciência de Dados.

🔗 Notebook no Google Colab: [https://colab.research.google.com/drive/1eC4ZWODTK7h7MoZZhWW-p7M6I-QdXyEJ?usp=sharing]

---

## Descrição

Este projeto implementa e compara diferentes modelos de **Machine Learning supervisionado** para resolver um problema de **classificação binária**: prever se um paciente possui (`1`) ou não possui (`0`) doença cardíaca, com base em atributos clínicos e resultados de exames laboratoriais.

O foco não é apenas obter o melhor modelo possível, mas construir uma solução **coerente, reproduzível, bem documentada e tecnicamente justificada**, seguindo boas práticas de Ciência de Dados: pipelines de pré-processamento, prevenção de vazamento de dados (*data leakage*), validação cruzada, otimização de hiperparâmetros e comparação justa entre modelos.

---

## Contexto

Doenças cardiovasculares estão entre as principais causas de mortalidade no mundo, e o diagnóstico precoce é fundamental para o tratamento adequado. A relação entre atributos clínicos (idade, colesterol, frequência cardíaca, resultados de eletrocardiograma, entre outros) e a ocorrência de doença cardíaca é complexa e difícil de mapear por regras determinísticas simples, o que torna o problema adequado para abordagens de Machine Learning capazes de identificar padrões nos dados.

O modelo resultante deve ser entendido como uma **ferramenta de apoio ao diagnóstico**, e não como substituto da avaliação médica.

---

## Objetivos

- Construir e avaliar modelos de classificação binária para prever a presença de doença cardíaca.
- Comparar um modelo baseline ingênuo com modelos candidatos mais elaborados.
- Aplicar boas práticas de pré-processamento, evitando vazamento de dados entre treino e teste.
- Realizar otimização de hiperparâmetros em pelo menos um modelo.
- Avaliar os modelos com métricas adequadas ao contexto clínico (priorizando a redução de falsos negativos).
- Selecionar o modelo final de forma coerente com o critério de sucesso definido (F1-score).
- Documentar de forma clara e reprodutível todas as decisões técnicas do projeto.

---

## Dataset

| Item | Descrição |
|---|---|
| **Nome** | Statlog Heart Disease Dataset |
| **Origem** | Kaggle / UCI Machine Learning Repository |
| **Link** | https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog |
| **Carregamento** | Via URL pública (raw) de repositório GitHub, sem necessidade de upload manual, login ou chave de API |
| **Registros** | 270 instâncias |
| **Atributos** | 13 atributos preditores + 1 variável-alvo (14 colunas no total) |
| **Valores ausentes** | Nenhum |
| **Duplicatas** | Nenhuma |
| **Balanceamento do target** | ≈ 55,5% classe 0 (ausência) / ≈ 44,5% classe 1 (presença) — considerado balanceado |

### Variável-alvo

- **`target`**: variável categórica binária — `1` indica presença de doença cardíaca, `0` indica ausência.

### Principais variáveis (atributos preditores)

| Coluna | Tipo | Descrição |
|---|---|---|
| `age` | numérica | Idade do paciente em anos |
| `sex` | categórica | Sexo do paciente (1 = masculino; 0 = feminino) |
| `cp` | categórica | Tipo de dor no peito (1 a 4) |
| `trestbps` | numérica | Pressão arterial em repouso (mm Hg) |
| `chol` | numérica | Colesterol sérico (mg/dl) |
| `fbs` | categórica | Açúcar no sangue em jejum > 120 mg/dl (1 = sim; 0 = não) |
| `restecg` | categórica | Resultados eletrocardiográficos em repouso (0, 1, 2) |
| `thalach` | numérica | Frequência cardíaca máxima atingida |
| `exang` | categórica | Angina induzida por exercício (1 = sim; 0 = não) |
| `oldpeak` | numérica | Depressão de ST induzida pelo exercício |
| `slope` | categórica | Inclinação do segmento ST de pico do exercício |
| `ca` | numérica | Número de vasos principais (0–3) coloridos por fluoroscopia |
| `thal` | categórica | Tipo de defeito (3 = normal; 6 = fixo; 7 = reversível) |

**Restrições/limitações conhecidas:** amostra relativamente pequena (270 instâncias), o que exige cautela quanto a *overfitting* e à variância das métricas estimadas por um único *holdout*. Dados anonimizados, de repositório acadêmico (UCI), sem informações que permitam identificação individual.

---

## Tecnologias Utilizadas

- **Python** 3 [INFORMAÇÃO NECESSÁRIA — versão exata do Python utilizada no ambiente final de execução]
- **Jupyter Notebook** / **Google Colab**
- **Pandas** — manipulação e análise de dados
- **NumPy** — operações numéricas
- **Matplotlib** — visualização de dados
- **Seaborn** — visualização estatística (histogramas, boxplots, heatmap de correlação)
- **Scikit-learn** — pipelines, pré-processamento, modelos de classificação, métricas, validação cruzada e busca de hiperparâmetros
  - `train_test_split`, `StratifiedKFold`, `cross_val_score`, `RandomizedSearchCV`
  - `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`
  - `DummyClassifier`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report`
- **SciPy** (`scipy.stats.randint`) — definição do espaço de busca de hiperparâmetros
- **Joblib** — serialização do modelo final treinado

---

## Estrutura do Projeto

```
.
├── README.md
├── main.ipynb                      # Notebook principal com todo o pipeline de ML
├── data/
│   └── raw/
│       └── Heart_disease_statlog.csv   # Dataset (carregado via URL raw do GitHub)
└── modelo_final_coracao.pkl        # Artefato do modelo final serializado (gerado ao executar o notebook)
```
---

## Metodologia

O projeto segue o fluxo padrão de um pipeline de Machine Learning supervisionado, estruturado em um único notebook executável do início ao fim:

1. **Definição do problema** — descrição do problema, objetivo do modelo, tipo de tarefa (classificação binária), variável-alvo, hipóteses e critérios de sucesso (métrica principal: F1-score).
2. **Configuração do ambiente** — fixação de seed (`SEED = 42`) para reprodutibilidade, importação de bibliotecas.
3. **Carregamento dos dados** — leitura do CSV diretamente via URL pública (raw do GitHub).
4. **Análise exploratória (EDA)** — formato do dataset, tipos de dados, valores ausentes, duplicatas, dicionário de dados, distribuição do target, histogramas de variáveis numéricas, gráfico de barras de variável categórica, boxplots por classe do target e heatmap de correlação.
5. **Preparação dos dados** — definição de features e target, divisão treino/teste (80/20) com estratificação (`stratify=y`), realizada **antes** de qualquer transformação para evitar vazamento de dados.
6. **Pré-processamento** — pipeline com `ColumnTransformer`: imputação (mediana para numéricas, moda para categóricas) + `StandardScaler` para variáveis numéricas; imputação + `OneHotEncoder` para variáveis categóricas.
7. **Modelagem e treinamento** — definição de um baseline ingênuo (`DummyClassifier`) e quatro modelos candidatos, todos encapsulados em `Pipeline` (pré-processamento + modelo).
8. **Avaliação inicial** — avaliação por *holdout* (conjunto de teste) e checagem de estabilidade por validação cruzada estratificada (5 folds) no conjunto de treino.
9. **Otimização de hiperparâmetros** — `RandomizedSearchCV` (10 combinações, validação cruzada de 5 folds) aplicado ao `RandomForestClassifier`.
10. **Avaliação final** — seleção do modelo final com base no F1-score (métrica principal definida no início do projeto), relatório de classificação e matriz de confusão no conjunto de teste.
11. **Comparação final dos modelos** — tabela consolidando F1, acurácia, recall, AUC e tempo de treino de todos os modelos avaliados.
12. **Boas práticas e rastreabilidade** — registro de configuração do experimento, desempenho computacional, limitações e decisões técnicas.
13. **Conclusão** — síntese dos resultados, comparação com o baseline, limitações e trabalhos futuros.
14. **Salvamento de artefatos** — exportação do pipeline final treinado (`modelo_final_coracao.pkl`) via `joblib`.

---

## Modelos de Machine Learning

| Modelo | Papel no projeto |
|---|---|
| `DummyClassifier` (`strategy="most_frequent"`) | **Baseline ingênuo** — sempre prevê a classe majoritária; referência mínima para verificar se os demais modelos aprendem padrões reais. |
| `LogisticRegression` | Modelo linear, simples e interpretável; amplamente utilizado como referência em classificação binária na área da saúde. |
| `KNeighborsClassifier` (KNN) | Modelo baseado em distância; sensível à escala dos atributos, exige padronização. |
| `SVC` (Support Vector Machine) | Modelo baseado em margens/distância, com `probability=True` para suportar cálculo de AUC. |
| `RandomForestClassifier` | Modelo de *ensemble* (bagging de árvores de decisão), avaliado com hiperparâmetros padrão e, posteriormente, otimizado via busca aleatória. |

---

## Métricas de Avaliação

| Métrica | Justificativa de uso |
|---|---|
| **F1-score** | Métrica principal do projeto. Equilibra precisão e recall, sendo especialmente adequada em contextos médicos, nos quais tanto falsos positivos quanto falsos negativos têm custo relevante. |
| **Recall (Sensibilidade)** | Mede a capacidade de identificar corretamente pacientes com doença cardíaca; em diagnóstico médico, reduzir falsos negativos é prioritário. |
| **Acurácia** | Métrica complementar; não é utilizada isoladamente, pois pode ser enganosa mesmo em datasets razoavelmente balanceados. |
| **Precisão** | Utilizada em conjunto com o recall para o cálculo do F1-score. |
| **AUC (ROC)** | Mede a capacidade do modelo de separar as classes independentemente do limiar de decisão. |
| **Matriz de confusão / Classification Report** | Detalham os tipos de erro (falsos positivos e falsos negativos) do modelo final no conjunto de teste. |
| **Tempo de treinamento** | Registrado para avaliar o custo computacional de cada modelo. |

---

## Principais Resultados

| Modelo | F1-score | Acurácia | Recall | AUC | Tempo de treino |
|---|---:|---:|---:|---:|---:|
| Baseline (`DummyClassifier`) | 0,0000 | 0,5556 | 0,0000 | 0,5000 | 0,044 s |
| **Regressão Logística** | **0,8627** | 0,8704 | 0,9167 | 0,8986 | 0,123 s |
| SVM | 0,8400 | 0,8519 | 0,8750 | 0,8903 | 0,031 s |
| Random Forest (otimizado) | 0,8323 | 0,8300 | 0,8300 | — | 0,222 s |

- Todos os modelos candidatos superaram amplamente o baseline, confirmando que o problema exige — e se beneficia de — aprendizado de máquina.
- A **Regressão Logística** obteve o **melhor F1-score** entre todos os modelos avaliados, sendo selecionada como **modelo final**, em conformidade com o critério de sucesso definido no início do projeto (F1-score como métrica principal).
- O Random Forest, mesmo após otimização de hiperparâmetros via `RandomizedSearchCV` (`max_depth=3`, `min_samples_split=9`, `n_estimators=207`, F1 de validação ≈ 0,8323), não superou a Regressão Logística nessa métrica, embora permaneça uma alternativa competitiva.
- Tempos de treinamento de todos os modelos ficaram abaixo de 0,25 segundos, tornando o custo computacional irrelevante para este MVP.

*Nenhuma métrica adicional foi inventada; os valores acima refletem exatamente os resultados obtidos e registrados no notebook.*

---

## Como Executar o Projeto

### Pré-requisitos

- Python 3.x
- Jupyter Notebook ou Google Colab

### Passo a passo

1. **Clone o repositório:**
   ```bash
   git clone [INFORMAÇÃO NECESSÁRIA — URL do repositório GitHub]
   cd [INFORMAÇÃO NECESSÁRIA — nome da pasta do repositório]
   ```

2. **Instale as dependências:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
   ```
   [INFORMAÇÃO NECESSÁRIA — recomenda-se criar um arquivo `requirements.txt` com versões fixadas das bibliotecas para garantir reprodutibilidade exata]

3. **Execute o notebook:**
   - **Opção A — Google Colab:** abra `main.ipynb` diretamente no Google Colab e execute todas as células em sequência (`Ambiente de execução > Executar tudo`).
   - **Opção B — Jupyter local:**
     ```bash
     jupyter notebook main.ipynb
     ```
     e execute as células em ordem, do início ao fim.

4. **Dataset:** o notebook carrega o dataset automaticamente via URL pública (raw do GitHub), não sendo necessário download manual.

5. **Saída esperada:** ao final da execução, será gerado o arquivo `modelo_final_coracao.pkl`, contendo o pipeline completo (pré-processamento + modelo final) treinado.

---

## Estrutura do Fluxo do Projeto

```
Dados (CSV via URL) 
        │
        ▼
Análise Exploratória (EDA)
        │
        ▼
Divisão Treino/Teste (80/20, estratificada)
        │
        ▼
Pipeline de Pré-processamento
 (Imputação → Padronização / One-Hot Encoding)
        │
        ▼
Treinamento: Baseline + Modelos Candidatos
 (Regressão Logística, KNN, SVM, Random Forest)
        │
        ▼
Validação Cruzada (checagem de estabilidade)
        │
        ▼
Otimização de Hiperparâmetros (Random Forest)
        │
        ▼
Avaliação Final no Conjunto de Teste
 (métrica principal: F1-score)
        │
        ▼
Seleção do Modelo Final (Regressão Logística)
        │
        ▼
Serialização do Modelo (joblib)
```

---

## Limitações

- O conjunto de dados possui apenas **270 instâncias**, o que reduz a confiança estatística dos resultados e aumenta a variância das métricas estimadas por um único *holdout*.
- Pequenas variações no conjunto de teste podem produzir mudanças relevantes nas métricas de avaliação.
- A base **Statlog Heart Disease** pode não representar toda a diversidade de perfis clínicos encontrados em diferentes hospitais, regiões ou populações.
- O modelo não deve ser utilizado como ferramenta única de diagnóstico médico, especialmente em situações de emergência ou em populações pouco representadas na base original.
- Ainda existem falsos negativos no modelo final, um erro crítico em contexto clínico.

---

## Melhorias Futuras

- Ampliar a base de dados para aumentar a robustez estatística do modelo.
- Avaliar algoritmos de *boosting*, como **XGBoost**, **LightGBM** e **CatBoost**.
- Investigar arquiteturas de redes neurais, caso conjuntos de dados maiores se tornem disponíveis.
- Incorporar novas variáveis clínicas, demográficas e comportamentais (hábitos de vida, histórico familiar).
- Realizar validações externas com dados de diferentes hospitais ou populações.
- Repetir a divisão treino/teste com múltiplas seeds (ou validação cruzada repetida) para quantificar a variância das métricas reportadas.
- Adicionar técnicas de interpretabilidade de modelo (ex.: importância de atributos, SHAP) para reforçar a confiança clínica nas previsões.
- Preparar o pipeline para *deploy* (ex.: API de inferência) caso o modelo avance além da fase de MVP.

---

## Licença

MIT License

Copyright (c) 2026 Robson M. Cambraia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Autor

**Autor:** Robson Mendes Cambraia
