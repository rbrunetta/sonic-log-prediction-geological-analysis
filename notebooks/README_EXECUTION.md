# 📋 Guia de Execução - Comparação de Métodos de Cross-Validation

## 🎯 Objetivo

Comparar 3 métodos de otimização de hiperparâmetros para entender qual é mais adequado para o cenário LOWO (Leave-One-Well-Out):

1. **K-Fold (cv=50)** - Baseline (notebook 05a)
2. **GroupKFold (n_splits=10)** - Mantém poços separados (notebook 05b)
3. **LeaveOneGroupOut (cv=32)** - LOWO completo (notebook 05c)

---

## 📁 Notebooks Criados

### **05a_xgboost_lowo_kfold.ipynb**
- **Método:** RandomizedSearchCV com K-Fold tradicional (cv=50)
- **Tempo estimado:** ~1 hora
- **Descrição:** Baseline - otimização tradicional com K-Fold (mistura poços)
- **Salva em:** `results/xgboost/`

### **05b_xgboost_lowo_groupkfold.ipynb**
- **Método:** RandomizedSearchCV com GroupKFold(n_splits=10)
- **Tempo estimado:** 5-8 horas
- **Descrição:** Garante que cada fold de CV contém poços COMPLETOS separados
- **Salva em:** `results/xgboost_groupkfold/`

### **05c_xgboost_lowo_full.ipynb**
- **Método:** RandomizedSearchCV com LeaveOneGroupOut()
- **Tempo estimado:** 20-30 horas ⚠️
- **Descrição:** Cada poço é testado individualmente (32 folds)
- **Salva em:** `results/xgboost_lowo_cv/`
- **⚠️ AVISO:** Computacionalmente intensivo! Recomenda-se rodar overnight

### **06_comparison_cv_methods.ipynb**
- **Método:** Análise comparativa
- **Tempo estimado:** 5-10 minutos
- **Descrição:** Compara resultados dos 3 métodos
- **Pré-requisito:** Executar notebooks 05, 05b e 05c primeiro

---

## 🚀 Ordem de Execução

### **Passo 1: Executar notebook baseline (05a)**

**⏱️ Tempo: ~1 hora**

Este é o método baseline (K-Fold tradicional). Execute primeiro para ter a referência.

```bash
# Abrir notebook
jupyter notebook 05a_xgboost_lowo_kfold.ipynb
```

**Ou executar em background:**
```bash
nohup jupyter nbconvert --to notebook --execute \
    05a_xgboost_lowo_kfold.ipynb \
    --output 05a_xgboost_lowo_kfold_executed.ipynb \
    > kfold_log.txt 2>&1 &
```

---

### **Passo 2: Executar notebook GroupKFold (05b)**

**⏱️ Tempo: 5-8 horas**

```bash
# Abrir notebook
jupyter notebook 05b_xgboost_lowo_groupkfold.ipynb
```

**Ou executar em background (Linux/Mac):**
```bash
nohup jupyter nbconvert --to notebook --execute \
    05b_xgboost_lowo_groupkfold.ipynb \
    --output 05b_xgboost_lowo_groupkfold_executed.ipynb \
    > groupkfold_log.txt 2>&1 &
```

**Verificar progresso:**
```bash
tail -f groupkfold_log.txt
```

---

### **Passo 3: Executar notebook LOWO-CV (05c)**

**⚠️ ATENÇÃO: Este notebook demora 20-30 horas!**

**Recomendações:**
1. Execute em servidor ou deixe o computador ligado overnight
2. Use `screen` ou `tmux` para não perder a sessão
3. Monitore o uso de GPU/CPU

**Execução:**
```bash
# Opção 1: Usando screen (recomendado)
screen -S lowo_cv
jupyter notebook 05c_xgboost_lowo_full.ipynb
# Pressione Ctrl+A+D para desconectar
# Para reconectar: screen -r lowo_cv

# Opção 2: Em background
nohup jupyter nbconvert --to notebook --execute \
    05c_xgboost_lowo_full.ipynb \
    --output 05c_xgboost_lowo_full_executed.ipynb \
    > lowo_cv_log.txt 2>&1 &
```

**Monitorar:**
```bash
# Ver log
tail -f lowo_cv_log.txt

# Ver uso de GPU
watch -n 1 nvidia-smi

# Ver arquivos sendo criados
watch -n 10 "ls -lh results/xgboost_lowo_cv/"
```

---

### **Passo 4: Executar análise comparativa (06)**

**⏱️ Tempo: 5-10 minutos**

Após concluir os 3 notebooks anteriores:

```bash
jupyter notebook 06_comparison_cv_methods.ipynb
```

Este notebook irá:
- Carregar resultados dos 3 métodos
- Comparar hiperparâmetros obtidos
- Comparar performance (R², RMSE, MAE)
- Identificar poços que melhoraram/pioraram
- Gerar visualizações
- Exportar tabela comparativa

---

## 📊 Resultados Esperados

Após executar todos os notebooks, você terá:

```
results/
├── xgboost/                    # Método 1: K-Fold (cv=50)
│   ├── metrics/
│   │   └── lowo_xgboost_results.csv
│   ├── predictions/
│   │   └── lowo_xgboost_predictions.csv
│   └── models/
│       └── xgboost_best_params.json
│
├── xgboost_groupkfold/         # Método 2: GroupKFold (cv=10)
│   ├── metrics/
│   │   └── lowo_xgboost_results.csv
│   ├── predictions/
│   │   └── lowo_xgboost_predictions.csv
│   └── models/
│       └── xgboost_best_params.json
│
├── xgboost_lowo_cv/            # Método 3: LOWO-CV (cv=32)
│   ├── metrics/
│   │   └── lowo_xgboost_results.csv
│   ├── predictions/
│   │   └── lowo_xgboost_predictions.csv
│   └── models/
│       └── xgboost_best_params.json
│
└── cv_methods_comparison.csv   # Tabela comparativa final
```

---

## 🔍 O que Analisar

### **1. Hiperparâmetros mudaram?**
- Compare os arquivos `xgboost_best_params.json` dos 3 métodos
- Parâmetros importantes: `max_depth`, `learning_rate`, `n_estimators`

### **2. Performance melhorou?**
- Compare R² médio dos 3 métodos
- Identifique qual tem menor variabilidade (std)

### **3. Poços problemáticos melhoraram?**
- Liste os 6 poços com R² < 0.50 no método K-Fold
- Verifique se GroupKFold ou LOWO-CV melhoraram esses poços

### **4. Vale o custo computacional?**
- Diferença de R² médio justifica 20-30h de computação?
- GroupKFold (10 folds) é um bom compromisso?

---

## ❓ FAQ

**Q: Posso executar os 3 notebooks em paralelo?**
A: Não recomendado. Todos usam GPU e podem competir por recursos. Execute sequencialmente.

**Q: Quanto de RAM/GPU é necessário?**
A: RAM: ~16GB, GPU: ~4GB VRAM (RTX 3050 é suficiente)

**Q: E se o notebook travar ou der erro?**
A: Os notebooks salvam resultados progressivamente. Você pode continuar de onde parou modificando o loop LOWO.

**Q: Preciso executar o notebook 05c (LOWO-CV)?**
A: Não é obrigatório. GroupKFold (05b) já oferece separação por poços. LOWO-CV é o método mais rigoroso, mas opcional.

**Q: Posso reduzir n_iter do RandomizedSearchCV?**
A: Sim. Trocar `n_iter=100` por `n_iter=50` reduz tempo pela metade, mas pode encontrar hiperparâmetros subótimos.

---

## 📝 Notas Importantes

1. **GPU:** Certifique-se que `device='cuda'` está ativado no XGBoost
2. **Memória:** Notebooks salvam DataFrames grandes. Monitore uso de RAM
3. **Backup:** Faça backup dos resultados do notebook 05 antes de começar
4. **Logs:** Salve logs de execução para análise posterior

---

## 🎯 Próximos Passos (Após Comparação)

Após analisar os resultados:

1. **Escolher melhor método** baseado em:
   - Rigor metodológico
   - Performance (R² médio)
   - Custo computacional

2. **Aplicar método escolhido** para:
   - LightGBM (notebook 07)
   - HistGradientBoosting (notebook 08)
   - RandomForest (notebook 09)

3. **Investigar poços problemáticos** que não melhoraram

---

## 📞 Contato

Se tiver dúvidas durante a execução, documente:
- Mensagem de erro completa
- Célula que deu erro
- Estado da execução (qual fold estava rodando)

---

**Boa sorte com as execuções!** 🚀

*Estimativa total de tempo: 25-40 horas*
