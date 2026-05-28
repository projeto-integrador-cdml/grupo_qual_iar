<div align="center">

# 🌬️ MotoAR

### Monitoramento e Previsão da Qualidade do Ar em Brasília

*Projeto Integrador — análise da poluição atmosférica no DF combinando dados oficiais do INMET com leituras de sensores IQAir, com modelo preditivo de PM2.5 e dashboard interativo.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![XGBoost](https://img.shields.io/badge/Modelo-XGBoost-EB6E4B?style=flat-square)
![Status](https://img.shields.io/badge/status-entrega-success?style=flat-square)

</div>

---

## 📖 Sobre o projeto

O **MotoAR** investiga a qualidade do ar no Distrito Federal a partir de duas
fontes complementares:

- **INMET** — estações automáticas oficiais (8.280 registros válidos em 2025);
- **IQAir** — 4 sensores comunitários espalhados por Brasília (36.301 leituras).

O fluxo completo cobre **ingestão → limpeza → EDA → modelagem → visualização**,
entregando um relatório técnico em PDF, um dashboard web responsivo e um
modelo preditivo de PM2.5.

## ✨ Destaques

| | |
|---|---|
| 🧹 **Pipeline de dados** | Limpeza, validação de status e tratamento de outliers em duas fontes heterogêneas |
| 📊 **EDA aprofundada** | Sazonalidade, padrões horários, heatmap hora × mês, efeito da chuva, correlações |
| 🤖 **Modelo preditivo** | XGBoost (com fallback para GradientBoosting) — **MAE 2.64 µg/m³, R² 0.834** |
| 🖥️ **Dashboard React** | Gráficos interativos com Recharts, modo claro/escuro e visualização desktop/mobile |
| 📄 **Relatório em PDF** | Exportação automática do relatório completo direto do dashboard via `html2pdf.js` |
| 🔁 **Reprodutível** | Dados brutos, dados limpos, modelo `.pkl` e JSON agregado versionados no repo |

## 👥 Integrantes

| Nome | |
|------|---|
| Rodrigo Lins | |
| Vitor Nascimento | |
| Alessandra | |
| Giovanna | |
| Rafael M. | |

## 🧱 Arquitetura

```
┌──────────────────┐    ┌──────────────────┐
│  INMET .xlsx     │    │  IQAir .csv      │
│  (estações 2025) │    │  (4 sensores DF) │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
           ┌──────────────────────┐
           │  motoar_pipeline.py  │   limpeza, status, clipes,
           │  build_data.py       │   features, agregações
           └──────────┬───────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  data/clean/   xgb_model.pkl   data_export.json
  (parquet/csv)  (modelo)      (front-ready)
                                   │
                                   ▼
                         ┌──────────────────┐
                         │  motoar-web/     │
                         │  React + Vite    │
                         │  Recharts + PDF  │
                         └──────────────────┘
```

## 📂 Estrutura do repositório

```
projeto-integrador-motoar/
├── README.md
├── ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx   # dados brutos INMET
├── iqair_data.csv                                 # dados brutos IQAir
├── motoar_pipeline.py                             # pipeline de limpeza/preparação
├── build_data.py                                  # gera data_export.json para o front
├── motoar_app.py / motoar_app2.py                 # apps de análise
├── data_export.json                               # agregações usadas pelo dashboard
├── motoair_app.html                               # protótipo HTML standalone
├── motoar_eda_lca.html                            # relatório de EDA + LCA
├── motoar_relatorio.html                          # relatório final em HTML
├── data/
│   ├── clean/
│   │   ├── iqair_clean.csv
│   │   ├── iqair_clean.parquet
│   │   └── pipeline_report.json
│   ├── EDA + LCA + Cruzamentos.pdf
│   ├── motoar_relatorio_completo.pdf
│   └── xgb_model.pkl                              # modelo treinado
└── motoar-web/                                    # dashboard React + Vite
    ├── src/
    │   ├── App.tsx
    │   ├── data.json
    │   └── ...
    ├── public/
    └── package.json
```

## 🗂️ Dados utilizados

| Fonte | Registros | Cobertura | Variáveis principais |
|-------|-----------|-----------|----------------------|
| **INMET** — estações automáticas 2025 | 8.280 (válidos) | DF, dados horários | PM2.5, PM10, NO₂, CO, O₃, SO₂, chuva, status |
| **IQAir** | 36.301 | 4 sensores em Brasília<sup>*</sup> | AQI, PM2.5, temperatura, umidade, vento |

<sup>*</sup> Sensores: **Brasília**, **Escola 115 Norte**, **Finatec**, **UnB Odisseia Gama**.

A limpeza descarta registros com `Status PM25` inválido, aplica clipes de
faixa para reduzir outliers de sensores e classifica cada registro por
estação climática (**Chuva** / **Seca** / **Transição**).

## 🤖 Modelo preditivo

Treinado em `build_data.py` sobre os dados INMET tratados.

- **Algoritmo:** XGBoost — fallback automático para `GradientBoostingRegressor`.
- **Split:** 80/20 — **6.624 registros** treino · **1.656** teste.
- **Métricas atuais:**

| Métrica | Valor |
|---------|------:|
| MAE | **2.638 µg/m³** |
| R² | **0.834** |

**Features utilizadas (11):**

```
hour_sin · hour_cos · month_sin · month_cos · is_dry
pm25_lag1 · pm25_lag3 · pm25_roll3 · rain_acc6 · no2 · co
```

A importância das features e amostras de previsão vs real ficam
serializadas em `data_export.json` e são exibidas no dashboard.

## 🚀 Como executar

### 0. Pré-requisitos

| Ferramenta | Versão mínima | Para que serve |
|------------|---------------|----------------|
| **Python** | 3.10+ | Pipeline de dados, modelo e apps Streamlit |
| **Node.js** | 18+ | Dashboard React (Vite) |
| **npm** | 9+ | Vem com o Node |
| **Git** | qualquer recente | Clonar o repositório |

> Verifique no terminal: `python3 --version`, `node --version`, `npm --version`.

### 1. Clonar o repositório

```bash
git clone https://github.com/rodrigolinss/projeto-integrador-motoar.git
cd projeto-integrador-motoar
```

### 2. Pipeline e modelo (Python)

#### 2.1. Criar ambiente virtual *(recomendado)*

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 2.2. Instalar dependências

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost openpyxl pyarrow
```

> Se quiser rodar também os apps Streamlit (`motoar_app.py` / `motoar_app2.py`),
> instale também: `pip install streamlit plotly reportlab`.

#### 2.3. Rodar o pipeline de limpeza

O pipeline lê os dois CSV/XLSX brutos e gera os arquivos limpos em `data/clean/`.

```bash
# o nome do arquivo INMET no repo tem espaços — passe explicitamente:
python motoar_pipeline.py --inmet "ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx"
```

Outras opções úteis:

```bash
python motoar_pipeline.py --fmt csv          # salva CSV em vez de Parquet
python motoar_pipeline.py --skip-cross       # pula validação cruzada
python motoar_pipeline.py --out saida_custom # diretório de saída diferente
```

Saídas em `data/clean/`:

- `iqair_clean.parquet` (ou `.csv`)
- `inmet_clean.parquet` (ou `.csv`)
- `pipeline_report.json` — resumo da execução

#### 2.4. Gerar o JSON do dashboard + treinar o modelo

```bash
python build_data.py
```

Isso produz **`data_export.json`** na raiz do projeto, com todas as
agregações (séries horárias, mensais, heatmap, correlações, histogramas)
e treina o modelo. As métricas reportadas no console devem bater com:

```
✓ IQAir processado: 36.301 registros
✓ INMET processado: 8.280 registros
✓ Modelo treinado: MAE=2.638, R²=0.834
```

### 3. Dashboard web (React + Vite)

#### 3.1. Atualizar o JSON consumido pelo front *(quando rodar o pipeline de novo)*

O `App.tsx` importa `motoar-web/src/data.json`. Depois de rodar
`build_data.py`, copie o arquivo gerado para dentro do front:

**macOS / Linux**
```bash
cp data_export.json motoar-web/src/data.json
```

**Windows (PowerShell)**
```powershell
Copy-Item data_export.json motoar-web\src\data.json -Force
```

> Se você só quer ver o dashboard com os dados que já estão versionados,
> **pule este passo** — o `data.json` já está commitado no repo.

#### 3.2. Instalar dependências e rodar

```bash
cd motoar-web
npm install
npm run dev
```

Abra **http://localhost:5173** no navegador.

| Comando | O que faz |
|---------|-----------|
| `npm run dev` | Servidor de desenvolvimento com HMR |
| `npm run build` | Build de produção em `motoar-web/dist/` |
| `npm run preview` | Serve o build localmente para testar |
| `npm run lint` | Roda ESLint no código |

Para gerar o **relatório em PDF**, abra o dashboard e clique em
**Exportar PDF** no canto superior — o `html2pdf.js` renderiza a página
inteira em PDF A4.

### 4. (Opcional) Apps Streamlit

Versões alternativas de análise interativa, com Plotly:

```bash
pip install streamlit plotly reportlab
streamlit run motoar_app.py      # ou motoar_app2.py
```

Acesse **http://localhost:8501**.

### ✅ Checklist rápido

- [ ] Python 3.10+ e Node 18+ instalados
- [ ] Repositório clonado
- [ ] `pip install` das dependências Python
- [ ] `python motoar_pipeline.py --inmet "ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx"`
- [ ] `python build_data.py`
- [ ] (se rodou o pipeline) `cp data_export.json motoar-web/src/data.json`
- [ ] `cd motoar-web && npm install && npm run dev`
- [ ] Abrir **http://localhost:5173** 🎉

### 🛟 Troubleshooting

| Problema | Solução |
|----------|---------|
| `FileNotFoundError: ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx` | Use `--inmet "ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx"` (o arquivo no repo tem espaços) |
| `ModuleNotFoundError: No module named 'xgboost'` | `pip install xgboost` — sem ele o `build_data.py` cai no `GradientBoosting` |
| `Engine 'pyarrow' is required` ao salvar Parquet | `pip install pyarrow`, ou rode com `--fmt csv` |
| Vite não acha `data.json` | Confira que o arquivo existe em `motoar-web/src/data.json` |
| `npm install` lento ou trava | Apague `node_modules/` e `package-lock.json` e rode de novo |
| Porta 5173 ocupada | `npm run dev -- --port 3000` |

## 📈 Principais achados

- **Sazonalidade clara:** PM2.5 médio salta na estação seca (jul–out), com
  picos consistentes de fim de tarde nos sensores urbanos.
- **Chuva derruba a poluição:** dias com ≥20 mm de precipitação reduzem
  drasticamente a média de PM2.5.
- **Divergência entre fontes:** o AQI IQAir e o PM2.5 INMET seguem o mesmo
  padrão horário, mas com magnitudes distintas — útil para discutir
  representatividade espacial dos sensores.
- **Modelo robusto:** R² de 0.834 indica que componentes temporais (lags,
  hora, mês) e poluentes correlatos (NO₂, CO) explicam bem a variância de
  PM2.5.

## 📦 Entregáveis

- `data/EDA + LCA + Cruzamentos.pdf` — análise exploratória completa.
- `data/motoar_relatorio_completo.pdf` — relatório final consolidado.
- `motoar_relatorio.html` — versão web do relatório.
- `motoar-web/` — dashboard interativo.
- `data/xgb_model.pkl` — modelo treinado pronto para uso.

## 🛠️ Stack

**Backend / Dados:** Python · pandas · NumPy · scikit-learn · XGBoost · openpyxl
**Frontend:** React 19 · TypeScript · Vite · Recharts · lucide-react · html2pdf.js

## 📜 Licença

Uso acadêmico. Os dados brutos pertencem ao **INMET** e à **IQAir** e estão
incluídos apenas para fins de reprodução do trabalho.

---

<div align="center">

*Projeto Integrador · Brasília · 2025*

</div>
