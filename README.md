<div align="center">

#  MotoAR

### Monitoramento e Previsão da Qualidade do Ar em Brasília

*Projeto Integrador — Ciência de Dados e Machine Learning · UniCEUB · 2026*

🌐 **[https://projeto-integrador-motoar.vercel.app](https://projeto-integrador-motoar.vercel.app)**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/Modelo-XGBoost-EB6E4B?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Status](https://img.shields.io/badge/status-entrega-success?style=flat-square)

</div>

---

## 👥 Equipe

| Nome | GitHub |
|------|--------|
| Alessandra Souza Gonçalves | [@alessandra2307](https://github.com/alessandra2307) |
| Vitor Nascimento | [@VitorFranco721](https://github.com/VitorFranco721) |
| Rodrigo Lin | [@rodrigolinss](https://github.com/rodrigolinss) |
| Giovanna | [@giovannafbarbosa](https://github.com/giovannafbarbosa) |
| Rafael Mascarenhas | [@Rafafelbrown](https://github.com/Rafafelbrown) |
**Orientador:** Prof. Weslley Rodrigues  
**Curso:** Ciência de Dados e Machine Learning  
**Instituição:** Centro Universitário de Brasília — UniCEUB  
**Ano:** 2026

---

## Sobre o projeto

O **MotoAR** investiga a qualidade do ar no Distrito Federal a partir de duas fontes complementares:

- **INMET** — estações automáticas oficiais (8.280 registros válidos em 2025)
- **IQAir** — 4 sensores comunitários espalhados por Brasília (36.063 leituras)

O projeto entrega um **pipeline de dados completo com arquitetura medalhão (Bronze → Silver → Gold)**, rastreamento de experimentos com MLflow, validação de qualidade dos dados e um dashboard interativo com modelo preditivo de PM2.5.

---

## Destaques

| | |
|---|---|
| **Pipeline Medalhão** | Arquitetura Bronze → Silver → Gold com orquestrador e validação de qualidade |
| **MLflow** | Rastreamento de experimentos, métricas e artefatos do modelo |
| **Modelo preditivo** | XGBoost — **MAE 2.604 µg/m³, R² 0.84** |
| **Dashboard Streamlit** | Gráficos interativos, modo claro/escuro, exportação PDF |
| **Dashboard React** | Gráficos com Recharts, responsivo, exportação PDF via html2pdf.js |
| **Testes automatizados** | Suíte de testes cobrindo pipeline e modelo |
| **Coleta automática** | Script de coleta via API AQICN para alimentar a camada Bronze |

---

## Arquitetura Medalhão

```
┌─────────────────────┐    ┌─────────────────────┐
│  INMET .xlsx        │    │  IQAir .csv         │
│  (estações 2025)    │    │  (4 sensores DF)    │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └─────────────┬────────────┘
                         ▼
              ┌─────────────────────┐
              │  BRONZE             │  bronze_ingest.py
              │  data/bronze/       │  cópia bruta + metadados
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  SILVER             │  silver_transform.py
              │  data/silver/       │  limpeza, validação,
              └──────────┬──────────┘  features temporais
                         ▼
              ┌─────────────────────┐
              │  GOLD               │  gold_build.py + MLflow
              │  data/gold/         │  agregações + modelo
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Dashboard          │
              │  Streamlit / React  │
              └─────────────────────┘
```

---

## Estrutura do repositório

```
grupo_qual_iar/
├── README.md
├── ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx   # dados brutos INMET
├── iqair_data.csv                                 # dados brutos IQAir
├── motoar_pipeline.py                             # pipeline de limpeza original
├── build_data.py                                  # gera data_export.json
├── motoar_app.py                                  # dashboard Streamlit principal
├── coletor.py                                     # coleta automática via API AQICN
├── env.example.txt                                # template de variáveis de ambiente
├── data_export.json                               # agregações para o dashboard
│
├── pipeline/                                      # ← NOVO: pipeline medalhão
│   ├── bronze/
│   │   └── bronze_ingest.py                       # ingestão sem esquema
│   ├── silver/
│   │   └── silver_transform.py                    # limpeza e validação
│   ├── gold/
│   │   └── gold_build.py                          # agregações + MLflow
│   ├── quality/
│   │   └── quality_check.py                       # validação de qualidade
│   └── orchestration/
│       └── run_pipeline.py                        # orquestrador ponta a ponta
│
├── tests/                                         # ← NOVO: testes automatizados
│   └── test_motoar.py
│
├── data/
│   ├── raw/                                       # dados brutos (bronze)
│   │   ├── iqair_data.csv
│   │   └── ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx
│   ├── silver/                                    # dados limpos
│   ├── gold/                                      # dados prontos + modelo
│   ├── EDA + LCA + Cruzamentos.pdf
│   └── motoar_relatorio_completo.pdf
│
└── motoar-web/                                    # dashboard React + Vite
    ├── src/
    │   ├── App.tsx
    │   ├── data.json
    │   └── ...
    └── package.json
```

---

## Dados utilizados

| Fonte | Registros | Cobertura | Variáveis principais |
|-------|-----------|-----------|----------------------|
| **INMET** — estações automáticas 2025 | 8.280 válidos | DF, dados horários | PM2.5, PM10, NO₂, CO, O₃, SO₂, chuva |
| **IQAir** | 36.063 | 4 sensores em Brasília | AQI, PM2.5, temperatura, umidade, vento |

Sensores IQAir: **Brasília**, **Escola 115 Norte**, **Finatec**, **UnB Odisseia Gama**

---

## Modelo preditivo

- **Algoritmo:** XGBoost (fallback automático para GradientBoostingRegressor)
- **Rastreamento:** MLflow — métricas, parâmetros e artefatos registrados automaticamente
- **Split temporal:** 80% treino (passado) / 20% teste (futuro) — sem data leakage

| Métrica | Valor |
|---------|------:|
| MAE | **2.604 µg/m³** |
| R² | **0.84** |

**Features (11):** `hour_sin · hour_cos · month_sin · month_cos · is_dry · pm25_lag1 · pm25_lag3 · pm25_roll3 · rain_acc6 · no2 · co`

---

## Como executar

### Pré-requisitos

| Ferramenta | Versão mínima |
|------------|---------------|
| Python | 3.10+ |
| Node.js | 18+ |
| Git | qualquer |

### 1. Clonar o repositório

```bash
git clone https://github.com/projeto-integrador-cdml/grupo_qual_iar.git
cd grupo_qual_iar
```

### 2. Criar ambiente virtual e instalar dependências

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost openpyxl pyarrow streamlit plotly mlflow python-dotenv requests
```

### 3. Rodar o pipeline medalhão completo

```bash
cd MotoAR
python pipeline\orchestration\run_pipeline.py --fmt csv
```

Isso executa automaticamente em sequência:

```
✅ Bronze  → copia dados brutos para data/bronze/
✅ Silver  → limpeza e validação dos dados
✅ Gold    → agregações + treinamento do modelo com MLflow
✅ Quality → validação de qualidade (16 expectativas)
```

### 4. Rodar o dashboard Streamlit

```bash
streamlit run motoar_app.py
```

Acesse **http://localhost:8501**

### 5. (Opcional) Dashboard React

```bash
cd motoar-web
npm install
npm run dev
```

Acesse **http://localhost:5173**

### Checklist rápido

- [ ] Python 3.10+ instalado
- [ ] Repositório clonado
- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas
- [ ] `python pipeline\orchestration\run_pipeline.py --fmt csv`
- [ ] `streamlit run motoar_app.py`
- [ ] Abrir **http://localhost:8501**

---

## Configurar coleta automática de dados (opcional)

O projeto inclui um coletor automático via API AQICN para alimentar a camada Bronze com dados novos:

1. Acesse **https://aqicn.org/data-platform/token** e cadastre seu email para obter o token gratuito
2. Copie o arquivo de exemplo: `cp env.example.txt .env`
3. Abra o `.env` e preencha: `AQICN_TOKEN=seu_token_aqui`
4. Rode o coletor:

```bash
python coletor.py           # coleta uma vez
python coletor.py --loop    # coleta a cada 1 hora
python coletor.py --status  # mostra estatísticas
```

---

## Principais achados

- **Sazonalidade clara:** PM2.5 médio salta na estação seca (jul–out), com pico em setembro (22.2 µg/m³)
- **Chuva reduz poluição:** dias com ≥20 mm de precipitação reduzem drasticamente a média de PM2.5
- **Padrão horário:** pico de poluição entre 20h–22h em todas as estações climáticas
- **Modelo robusto:** R² de 0.84 com lags temporais e poluentes correlatos (NO₂, CO)
- **16/16 validações de qualidade** aprovadas no pipeline medalhão

---

## Entregáveis

- `data/EDA + LCA + Cruzamentos.pdf` — análise exploratória completa
- `data/motoar_relatorio_completo.pdf` — relatório final consolidado
- `pipeline/` — pipeline medalhão com Bronze, Silver, Gold e Quality
- `tests/` — testes automatizados
- Dashboard Streamlit (`motoar_app.py`)
- Dashboard React (`motoar-web/`)

---

## Stack

**Dados:** Python · pandas · NumPy · scikit-learn · XGBoost · MLflow · openpyxl · pyarrow  
**Dashboard:** Streamlit · Plotly · React 19 · TypeScript · Vite · Recharts · html2pdf.js  
**Qualidade:** Testes automatizados · Validação de expectativas estilo Great Expectations

---

## Licença

Uso acadêmico. Os dados brutos pertencem ao **INMET** e à **IQAir** e estão incluídos apenas para fins de reprodução do trabalho.

---

<div align="center">

*Projeto Integrador · Ciência de Dados e Machine Learning · UniCEUB · Brasília · 2026*  
*Orientador: Prof. Weslley Rodrigues*

</div>
