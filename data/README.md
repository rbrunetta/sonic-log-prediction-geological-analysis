# Data Directory

## 📂 Estrutura

- `raw/` - Dados originais, nunca modificados
- `processed/` - Dados processados e prontos para uso

## 📊 Datasets

### raw/
- Coloque aqui os dados brutos originais
- **Nunca modifique esses arquivos!**

### processed/
- `wells.csv` - Dados sem filtro IQR aplicado
- `wells_iqr.csv` - Dados com filtro IQR aplicado

## 🔄 Processamento

Para gerar os dados processados a partir dos raw, veja:
- `notebooks/01_exploratory_analysis.ipynb`

## 📝 Descrição dos Dados

[Descreva aqui suas features:]

- **DEPT**: Profundidade (m)
- **GR**: Raios Gama (API)
- **RHOB**: Densidade (g/cm³)
- **NPHI**: Porosidade Neutrônica (fração)
- **DT**: Perfil Sônico - TARGET (µs/ft)
- ... [adicione outras features]

## ⚠️ Observações

- Os dados com IQR têm lacunas devido à remoção de outliers
- Para detalhes sobre o preprocessamento, consulte a documentação
