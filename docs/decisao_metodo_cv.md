# Decisão Metodológica: Método de Cross-Validation para Otimização de Hiperparâmetros

**Data:** Dezembro 2025  
**Contexto:** Projeto de PhD — Predição de Sonic Log (DT), Bacia Sergipe-Alagoas  
**Notebook de origem:** `05_comparison_cv_methods.ipynb` (arquivado)

---

## Problema

Durante a otimização de hiperparâmetros via `RandomizedSearchCV`, três estratégias de cross-validation foram consideradas. A escolha do método impacta diretamente a validade dos hiperparâmetros selecionados e a integridade da avaliação do modelo.

---

## Métodos Avaliados

### Método 1 — K-Fold (cv=50)
Divide os dados em 50 partições aleatórias, sem considerar a estrutura de poços. Amostras de um mesmo poço podem aparecer simultaneamente em treino e teste dentro de um fold.

- **Vantagem:** Simples, amplamente utilizado na literatura
- **Problema crítico:** Data leakage entre poços — o modelo "vê" amostras do mesmo poço no treino que está tentando prever, inflando artificialmente as métricas de validação. Os hiperparâmetros selecionados são otimizados para interpolar dentro de um poço, não para generalizar entre poços diferentes

### Método 2 — GroupKFold (cv=10)
Divide os dados em 10 folds garantindo que todas as amostras de um mesmo poço fiquem inteiramente em treino **ou** em teste, nunca nos dois. Os grupos são os poços (`Well_Name`).

- **Vantagem:** Elimina o data leakage entre poços; avalia a capacidade de generalização para poços não vistos; custo computacional razoável (~12 min com XGBoost + GPU)
- **Alinhamento com LOWO:** Reproduz a lógica de generalização geológica que será avaliada formalmente no LOWO

### Método 3 — LeaveOneGroupOut (LOWO-CV, cv=32)
Usa o próprio LOWO como função de otimização: para cada uma das 100 combinações de hiperparâmetros testadas, executa os 32 folds completos do LOWO.

- **Vantagem:** Máximo rigor — os hiperparâmetros são selecionados exatamente para o cenário de avaliação final
- **Problema prático:** Custo computacional proibitivo (~40 min) para 100 iterações. Adicionalmente, os hiperparâmetros encontrados foram **idênticos** aos do GroupKFold (cv=10) até a décima quinta casa decimal, sugerindo que as 100 combinações aleatórias geradas pelo `random_state=42` são as mesmas nos dois casos — o espaço de busca é coberto da mesma forma

---

## Resultado da Comparação

A performance LOWO final (R² médio em 32 poços) foi **equivalente** entre os três métodos. A diferença entre GroupKFold e LOWO-CV foi inferior a 0.001 em R², dentro da margem de variação estatística normal entre poços.

A principal distinção foi o tempo computacional:

| Método | Tempo (tuning) | R² médio LOWO | Hiperparâmetros distintos? |
|---|---|---|---|
| K-Fold (cv=50) | ~5 min | ~0.69 | Sim (leakage) |
| GroupKFold (cv=10) | ~12 min | ~0.69 | Não |
| LOWO-CV (cv=32) | ~40 min | ~0.69 | Não (idêntico ao GroupKFold) |

---

## Decisão

**GroupKFold (cv=10) foi adotado como método padrão de otimização de hiperparâmetros** para todos os algoritmos do benchmark (XGBoost, LightGBM, HistGradientBoosting, RandomForest).

**Justificativas:**

1. **Rigor metodológico:** Elimina o data leakage entre poços, garantindo que os hiperparâmetros sejam selecionados para generalização geológica real e não para interpolação dentro de um poço

2. **Equivalência prática com LOWO-CV:** Os hiperparâmetros encontrados foram idênticos ao método mais rigoroso (LOWO-CV), indicando que cv=10 é suficiente para cobrir o espaço de busca com `n_iter=100`

3. **Eficiência computacional:** Três vezes mais rápido que LOWO-CV sem perda de qualidade nos hiperparâmetros selecionados

4. **Alinhamento com a literatura:** GroupKFold é o método recomendado para dados com estrutura de grupos (pacientes, poços, estações) quando se deseja avaliar generalização entre grupos distintos

---

## Implicação para a Tese

A adoção do GroupKFold na fase de tuning reforça a coerência metodológica com o LOWO usado na avaliação final: ambos respeitam a independência entre poços como unidade geológica. Isso é particularmente relevante para a Bacia Sergipe-Alagoas, onde poços diferentes podem representar contextos deposicionais distintos.

O K-Fold tradicional, embora mais comum na literatura geral de machine learning, seria inadequado para este problema porque ignora a autocorrelação espacial entre amostras do mesmo poço.

---

## Referência

Notebook original: `notebooks/archive/05_comparison_cv_methods.ipynb`  
Resultado arquivado: `results/comparison/metrics/cv_methods_comparison.csv`
