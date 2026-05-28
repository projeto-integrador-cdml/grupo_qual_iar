"""
╔══════════════════════════════════════════════════════════╗
║  MotoAR — Análise de Qualidade do Ar para Motociclistas  ║
║  EDA · LCA · Cruzamentos · Modelo Preditivo              ║
╚══════════════════════════════════════════════════════════╝

Execute com:  streamlit run motoar_app.py
Dependências: pip install streamlit plotly pandas scikit-learn xgboost openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
import tempfile

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MotoAR — Análise de Dados",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── TEMA VISUAL ───────────────────────────────────────────────────────────────
COLORS = {
    # --- Originais ---
    "accent":     "#FF5722",
    "green":      "#2ECC71",
    "yellow":     "#F1C40F",
    "orange":     "#FF8C00",
    "red":        "#E74C3C",
    "blue":       "#3498DB",
    "teal":       "#1ABC9C",
    "bg":         "#0D0F12",
    "card":       "#1E2126",
    "txt":        "#F0F2F5",
    "txt2":       "#8A9099",

    # --- Novas Adições (Estrutura e Feedback) ---
    
    # Tons de Cinza para UI
    "border":     "#2D3139",  # Divisores sutis e bordas de inputs
    "hover":      "#2A2E35",  # Cor de fundo quando passa o mouse no card
    "overlay":    "rgba(0, 0, 0, 0.6)", # Para modais e fundos escurecidos
    
    # Variações de Texto
    "txt_disabled": "#565C66", # Texto desativado ou menos importante
    "txt_link":     "#5DADE2", # Um azul levemente mais claro para links no corpo do texto
    
    # Estados de Interação (Versões mais escuras das principais)
    "accent_dark":  "#E64A19", # Hover do botão principal
    "green_dark":   "#27AE60", # Hover de sucesso
    "red_dark":     "#C0392B", # Hover de erro/perigo
    
    # Cores de "Status Soft" (Para fundos de alertas/badges)
    "green_soft":   "rgba(46, 204, 113, 0.1)",
    "red_soft":     "rgba(231, 76, 60, 0.1)",
    "blue_soft":    "rgba(52, 152, 219, 0.1)",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#F0F2F5", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.15)"),
    colorway=[COLORS["accent"], COLORS["teal"], COLORS["blue"],
              COLORS["yellow"], COLORS["green"], COLORS["red"]],
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding: 1.5rem 2rem 3rem; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: -0.5px; }

/* metric cards */
[data-testid="metric-container"] {
    background: #1E2126;
    border: 0.5px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { color: #8A9099 !important; font-size: 11px !important; font-family: 'IBM Plex Mono'; text-transform: uppercase; letter-spacing: .07em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Barlow Condensed'; font-size: 28px !important; }

/* sidebar */
[data-testid="stSidebar"] { background: #141719 !important; border-right: 0.5px solid rgba(255,255,255,0.07); }
[data-testid="stSidebar"] * { color: #F0F2F5 !important; }

/* section tags */
.sec-tag { font-family: 'IBM Plex Mono'; font-size: 10px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
           color: #FF5722; border-left: 3px solid #FF5722; padding-left: 8px; margin-bottom: 4px; }
.sec-title { font-family: 'Barlow Condensed'; font-size: 32px; font-weight: 900; margin: 0 0 .25rem; }
.sec-sub { font-size: 13px; color: #8A9099; line-height: 1.6; margin-bottom: 1.5rem; }

/* insight pills */
.insight-pill { background: #1E2126; border: 0.5px solid rgba(255,255,255,0.1); border-radius: 10px;
                padding: 1rem 1.25rem; margin-bottom: .5rem; }
.insight-pill h4 { font-family: 'IBM Plex Mono'; font-size: 11px; text-transform: uppercase; color: #8A9099; margin: 0 0 4px; }
.insight-pill p { font-size: 12px; color: #F0F2F5; margin: 0; line-height: 1.5; }
.insight-pill span { font-family: 'Barlow Condensed'; font-size: 26px; font-weight: 700; display: block; margin-top: 4px; }

/* tables */
.styled-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono'; font-size: 11px; }
.styled-table th { background: #1E2126; color: #8A9099; padding: 8px 12px; text-align: left; text-transform: uppercase; letter-spacing: .06em; font-size: 10px; border-bottom: 1px solid rgba(255,255,255,.1); }
.styled-table td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,.05); color: #C0C4CC; }
.styled-table tr:hover td { background: rgba(255,255,255,.03); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 100px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
.b-red { background: rgba(231,76,60,.2); color: #E74C3C; }
.b-green { background: rgba(46,204,113,.2); color: #2ECC71; }
.b-yellow { background: rgba(241,196,15,.2); color: #F1C40F; }
.b-blue { background: rgba(52,152,219,.2); color: #3498DB; }
.b-orange { background: rgba(255,140,0,.2); color: #FF8C00; }

/* divider */
.hrdiv { border: none; border-top: 0.5px solid rgba(255,255,255,.08); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── CACHE DE DADOS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_iqair(uploaded=None):
    if uploaded:
        df = pd.read_csv(uploaded)
    elif os.path.exists("iqair_data.csv"):
        df = pd.read_csv("iqair_data.csv")
    else:
        return None
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["temperature"] = pd.to_numeric(df["temperature"].astype(str).str.replace("°C","").str.strip(), errors="coerce")
    df["humidity"]    = pd.to_numeric(df["humidity"].astype(str).str.replace("%","").str.strip(), errors="coerce")
    df["wind_speed"]  = pd.to_numeric(df["wind_speed"].astype(str).str.replace("km/h","").str.strip(), errors="coerce")
    df["hour"] = df["created_at"].dt.hour
    df["date"] = df["created_at"].dt.date
    df["month"] = df["created_at"].dt.month
    return df

@st.cache_data
def load_inmet(uploaded=None):
    path = None
    if uploaded:
        path = uploaded
    elif os.path.exists("ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx"):
        path = "ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx"
    if path is None:
        return None
    df = pd.read_excel(path, skiprows=2, header=0)
    df.rename(columns={"Data/Hora": "dt"}, inplace=True)
    df["dt"]    = pd.to_datetime(df["dt"])
    df["hour"]  = df["dt"].dt.hour
    df["month"] = df["dt"].dt.month
    df["date"]  = df["dt"].dt.date

    valid = df[df["Status PM25"] == "Ok"].copy()
    valid["pm25"] = pd.to_numeric(valid["PM25 (ug/m3)"], errors="coerce")
    valid["pm10"] = pd.to_numeric(valid["PM10 (ug/m3)"], errors="coerce")
    valid["no2"]  = pd.to_numeric(valid["NO2_ug/m3 (ug/m3)"], errors="coerce").clip(0, 300)
    valid["co"]   = pd.to_numeric(valid["CO_ppm (ppm)"], errors="coerce").clip(0, 20)
    valid["o3"]   = pd.to_numeric(valid["O3_ug/m3 (ug/m3)"], errors="coerce").clip(0, 500)
    valid["so2"]  = pd.to_numeric(valid["SO2_ug/m3 (ug/m3)"], errors="coerce").clip(0, 500)
    valid["rain"] = pd.to_numeric(valid["Rain (mm)"], errors="coerce").clip(lower=0)
    valid = valid[(valid["pm25"] >= 0) & (valid["pm25"] < 500)]
    valid["season"] = valid["month"].map({
        1:"🌧️ Chuva", 2:"🌧️ Chuva", 3:"🌧️ Chuva",
        4:"🌧️ Chuva", 5:"🌧️ Chuva", 6:"🌧️ Chuva",
        7:"🔥 Seca",  8:"🔥 Seca",  9:"🔥 Seca", 10:"🔥 Seca",
        11:"🍂 Transição", 12:"🍂 Transição"
    })
    return valid


def fig_layout(fig, height=350, margin=None):
    m = margin or dict(l=10, r=10, t=30, b=10)
    fig.update_layout(
        height=height, margin=m,
        **PLOTLY_TEMPLATE,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.1)"),
    )
    return fig

MONTH_LABELS = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
SEASON_COLORS = {"🌧️ Chuva": COLORS["blue"], "🔥 Seca": COLORS["red"], "🍂 Transição": COLORS["yellow"]}


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏍️ MotoAR")
    st.markdown("**Análise de Qualidade do Ar**")
    st.markdown("---")

    st.markdown("### 📁 Dados")
    up_iq    = st.file_uploader("IQAir CSV", type=["csv"], key="iq")
    up_inmet = st.file_uploader("INMET 2025 XLSX", type=["xlsx"], key="inmet")

    st.markdown("---")
    page = st.radio("Navegação", [
        "🏠 Visão Geral",
        "📋 LCA do Projeto",
        "🔬 EDA — INMET 2025",
        "📡 EDA — IQAir",
        "🔗 Cruzamentos",
        "🤖 Modelo Preditivo",
        "🏍️ Recomendação Gear",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#555;font-family:IBM Plex Mono;'>
    INMET CRAS Fercal · 2025<br>
    IQAir · 4 sensores · Mar 2026<br>
    Brasília-DF<br><br>
    
    </div>
    """, unsafe_allow_html=True)

# ─── CARREGA DADOS ───────────────────────────────────────────────────────────────
iq    = load_iqair(up_iq)
inmet = load_inmet(up_inmet)

# Fallback: tenta carregar dos paths padrão se não foram carregados
if iq is None and not up_iq:
    iq = load_iqair()
if inmet is None and not up_inmet:
    inmet = load_inmet()

def no_data(name):
    st.warning(f"⚠️ Arquivo **{name}** não carregado. Use o upload na barra lateral.")

def badge(txt, cls):
    return f'<span class="badge {cls}">{txt}</span>'


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════════
if "Visão Geral" in page:
    st.markdown('<div class="sec-tag">MotoAR · Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">VISÃO GERAL DO PROJETO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">App de qualidade do ar para motociclistas de Brasília. Combina dados INMET (estação fixa horária) com IQAir (4 sensores quasi-tempo real) para recomendar equipamentos e horários de saída.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Registros totais", "45.061", "INMET + IQAir")
    with c2: st.metric("Parâmetros", "18", "meteo + poluentes")
    with c3: st.metric("Sensores", "6", "2 INMET · 4 IQAir")
    with c4: st.metric("Cobertura", "14 meses", "Jan 2025 – Mar 2026")
    with c5: st.metric("Acurácia modelo*", "~82%", "estimada (XGBoost)")

    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Sazonalidade PM2.5 — CRAS Fercal 2025")
        monthly_pm = [5.1,7.13,9.49,6.07,4.2,5.84,17.77,18.3,22.25,18.82,12.12,9.44]
        colors_m   = [COLORS["blue"] if i<6 else COLORS["red"] if i<10 else COLORS["yellow"] for i in range(12)]
        fig = go.Figure(go.Bar(x=MONTH_LABELS, y=monthly_pm, marker_color=colors_m,
                                marker_line_width=0, text=[f"{v:.1f}" for v in monthly_pm],
                                textposition="outside", textfont_size=9))
        fig.update_layout(yaxis_title="µg/m³")
        st.plotly_chart(fig_layout(fig, 280), use_container_width=True)

    with col2:
        st.markdown("#### 🕐 Padrão horário — PM2.5 por estação")
        hourly_seca  = [28.09,24.93,21.19,19.01,17.46,16.61,16.33,17.27,21.22,23.51,18.0,9.91,8.24,7.52,7.97,7.06,8.91,12.58,16.86,26.69,34.0,33.43,31.91,30.72]
        hourly_chuva = [7.23,5.77,4.81,4.12,4.11,3.82,4.11,6.32,8.35,8.21,5.64,3.83,3.66,3.9,4.21,4.99,5.05,5.91,6.55,9.28,11.53,10.62,8.59,8.2]
        hourly_trans = [8.83,7.66,6.92,7.8,8.45,9.12,11.25,15.8,13.22,9.04,6.72,8.09,6.07,4.54,6.4,6.27,9.33,10.74,15.22,21.58,20.31,16.98,14.74,11.97]
        horas = list(range(24))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=horas, y=hourly_seca, name="🔥 Seca (Jul–Out)",
                                  line=dict(color=COLORS["red"], width=2.5), fill="tozeroy", fillcolor="rgba(231,76,60,0.08)"))
        fig.add_trace(go.Scatter(x=horas, y=hourly_trans, name="🍂 Transição",
                                  line=dict(color=COLORS["yellow"], width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=horas, y=hourly_chuva, name="🌧️ Chuva (Jan–Jun)",
                                  line=dict(color=COLORS["blue"], width=2), fill="tozeroy", fillcolor="rgba(52,152,219,0.06)"))
        fig.update_layout(xaxis_title="Hora", yaxis_title="PM2.5 µg/m³")
        st.plotly_chart(fig_layout(fig, 280), use_container_width=True)

    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
    st.markdown("#### ⚡ Achados principais")
    cols = st.columns(3)
    findings = [
        ("🔥", "Amplitude sazonal 5.3×", "PM2.5 vai de 4,2 µg/m³ (mai) a 22,3 µg/m³ (set). Temporada de queimadas domina tudo.", COLORS["red"]),
        ("🕐", "Dois picos diários", "7–9h (tráfego) e 19–21h (inversão térmica). Confirmado por INMET e IQAir de forma independente.", COLORS["orange"]),
        ("🌧️", "Chuva reduz PM2.5 em 34%", "Dias com chuva acumulada >5mm: PM2.5 médio de 7,9 vs 12,8 nos dias secos.", COLORS["blue"]),
        ("📍", "70% de diferença espacial", "UnB Gama tem AQI 70% maior que Finatec. Rota do motociclista importa tanto quanto hora.", COLORS["yellow"]),
        ("🔗", "NO2 correlaciona PM2.5", "r=0.47 entre NO2 e PM2.5. NO2 como proxy de combustão (tráfego + queimadas).", COLORS["teal"]),
        ("🪖", "65% das horas exigem filtro", "Na estação seca, 65% das horas noturnas ultrapassam o limiar OMS de 15 µg/m³.", COLORS["red"]),
    ]
    for i, (icon, title, txt, clr) in enumerate(findings):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="insight-pill" style="border-left: 3px solid {clr};">
                <h4>{icon} {title}</h4>
                <p>{txt}</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: LCA
# ══════════════════════════════════════════════════════════════════════════════════
elif "LCA" in page:
    st.markdown('<div class="sec-tag">01 — Ciclo de vida do projeto</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">LCA — MotoAR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Mapeamento completo do ciclo de vida dos dados: da coleta bruta ao MotoAR Score entregue ao motociclista.</div>', unsafe_allow_html=True)

    # Pipeline visual
    steps = [
        ("01", "Ingestão", ["IQAir API (free/premium)", "INMET estações automáticas", "Open-Meteo forecast", "Polling 5min (IQAir)", "Série horária (INMET)"]),
        ("02", "Limpeza", ["Filtro status 'Ok' INMET", "Clipping de outliers", "Imputação de nulos", "Normalização de unidades", "Detecção sensor falho"]),
        ("03", "Feature Eng.", ["Hora do dia (sin/cos)", "Flag estação seca/chuva", "Rolling avg 3h / 24h", "Chuva acumulada 6h", "Delta temp hora anterior"]),
        ("04", "Modelo", ["XGBoost / Random Forest", "Índice de Saída 0–100", "Previsão PM2.5 +6h", "Classificação de risco", "Recomendação de gear"]),
        ("05", "Entrega", ["Push notification", "Tela 'Agora' MotoAR", "Alerta queimadas Jul–Out", "Guia de equipamento", "Histórico pessoal"]),
    ]
    cols = st.columns(5)
    for col, (num, name, items) in zip(cols, steps):
        with col:
            items_html = "".join(f"<li>→ {it}</li>" for it in items)
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid rgba(255,255,255,0.1);border-radius:10px;padding:1rem;height:220px;">
                <div style="font-family:'IBM Plex Mono';font-size:9px;color:#FF5722;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">{num}</div>
                <div style="font-family:'Barlow Condensed';font-size:18px;font-weight:700;color:#F0F2F5;margin-bottom:10px;">{name}</div>
                <ul style="list-style:none;padding:0;margin:0;font-size:10px;color:#8A9099;line-height:1.8;">{items_html}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
    st.markdown("#### 📦 Inventário de dados")

    inventory = [
        ["INMET — CRAS Fercal", "Jan–Dez 2025", "8.760", "14 parâmetros", "1h", "Ativo ✅"],
        ["INMET — Estação Escola", "Jan–Dez 2025", "8.760", "8 parâmetros", "1h", "Ativo ✅"],
        ["IQAir — Brasilia (centro)", "Mar 2026", "9.853", "AQI, PM2.5, Meteo", "~1min", "Live 🟢"],
        ["IQAir — Escola 115 Norte", "Mar 2026", "8.766", "AQI, PM2.5, Meteo", "~1min", "Live 🟢"],
        ["IQAir — UnB Odisseia Gama", "Mar 2026", "9.053", "AQI, PM2.5, Meteo", "~1min", "Live 🟢"],
        ["IQAir — Finatec", "Mar 2026", "8.629", "AQI, PM2.5, Meteo", "~1min", "Live 🟢"],
        ["Open-Meteo (planejado)", "Futuro", "—", "Temp, Precip, Vento, UV", "1h", "Integrar 🔵"],
        ["INMET 2024 (planejado)", "Jan–Dez 2024", "8.784", "Mesmos parâmetros", "1h", "Pendente 🟡"],
    ]
    df_inv = pd.DataFrame(inventory, columns=["Fonte","Cobertura","Registros","Parâmetros","Freq.","Status"])
    st.dataframe(df_inv, use_container_width=True, hide_index=True,
                 column_config={"Registros": st.column_config.TextColumn(width="small"),
                                "Freq.": st.column_config.TextColumn(width="small")})

    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔑 APIs planejadas")
        apis = [
            ("IQAir Free", "Atual", "1000 req/dia", "green"),
            ("IQAir Premium", "Futuro", "Sem limite + histórico", "blue"),
            ("Open-Meteo", "Gratuita", "Sem chave, 16 dias forecast", "green"),
            ("OpenWeatherMap", "Opcional", "1000 req/dia free", "yellow"),
        ]
        for name, status, desc, clr in apis:
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid rgba(255,255,255,.1);border-radius:8px;padding:.75rem 1rem;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:12px;color:#F0F2F5;font-weight:500;">{name}</span><br>
                    <span style="font-size:10px;color:#8A9099;">{desc}</span>
                </div>
                <span class="badge b-{clr}">{status}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### ⚖️ Limitações identificadas")
        limits = [
            ("INMET — sensor temperatura inválido", "Valores negativos extremos filtrados"),
            ("IQAir — apenas mar/2026", "Série curta; ideal ≥6 meses"),
            ("INMET 2024 — não recebido ainda", "Reduce capacidade de treino"),
            ("Cobertura espacial limitada", "6 sensores para toda Brasília"),
            ("PM2.5 IQAir em escala AQI", "Não diretamente comparável com INMET µg/m³"),
        ]
        for title, desc in limits:
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid rgba(231,76,60,.2);border-radius:8px;padding:.75rem 1rem;margin-bottom:6px;">
                <span style="font-size:11px;color:#E74C3C;font-weight:600;">⚠ {title}</span><br>
                <span style="font-size:10px;color:#8A9099;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("#### 🎯 Próximos passos LCA")
        steps_next = [
            ("1", "Coletar INMET 2024", "Dobra o dataset de treino"),
            ("2", "Integrar Open-Meteo", "Previsão 16h para alertas antecipados"),
            ("3", "Treinar XGBoost", "Features de EDA + cruzamentos"),
            ("4", "Deploy API FastAPI", "Endpoint /score para o app"),
            ("5", "CI/CD de modelo", "Re-treino mensal automático"),
        ]
        for num, step, desc in steps_next:
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid rgba(255,255,255,.07);border-radius:8px;padding:.75rem 1rem;margin-bottom:6px;display:flex;gap:10px;align-items:flex-start;">
                <span style="font-family:'Barlow Condensed';font-size:20px;font-weight:700;color:#FF5722;line-height:1;">{num}</span>
                <div><span style="font-size:12px;color:#F0F2F5;">{step}</span><br><span style="font-size:10px;color:#8A9099;">{desc}</span></div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: EDA INMET
# ══════════════════════════════════════════════════════════════════════════════════
elif "INMET" in page:
    st.markdown('<div class="sec-tag">02 — Análise exploratória</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">EDA — INMET 2025</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Estação CRAS Fercal · Série horária completa Jan–Dez 2025 · 14 parâmetros ambientais</div>', unsafe_allow_html=True)

    if inmet is None:
        no_data("INMET 2025 XLSX")
    else:
        # Filtros sidebar
        sel_months = st.multiselect("Filtrar meses", MONTH_LABELS, default=MONTH_LABELS,
                                    format_func=lambda x: x)
        sel_idx = [MONTH_LABELS.index(m)+1 for m in sel_months] if sel_months else list(range(1,13))
        df = inmet[inmet["month"].isin(sel_idx)].copy()

        # Métricas
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.metric("Registros válidos PM2.5", f"{len(df):,}")
        with c2: st.metric("PM2.5 médio", f"{df['pm25'].mean():.1f} µg/m³")
        with c3: st.metric("PM2.5 máximo", f"{df['pm25'].max():.1f} µg/m³", delta="pico queimadas", delta_color="inverse")
        with c4: st.metric("Mediana (P50)", f"{df['pm25'].median():.1f} µg/m³")
        with c5: st.metric("Desvio padrão", f"{df['pm25'].std():.1f} µg/m³")

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### PM2.5 mensal — média, P75, máximo")
            monthly = df.groupby("month")["pm25"].agg(["mean","median",
                lambda x: x.quantile(.75), "max"]).reset_index()
            monthly.columns = ["month","mean","median","p75","max"]
            monthly["month_name"] = monthly["month"].apply(lambda m: MONTH_LABELS[m-1])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["mean"],
                name="Média", marker_color=[COLORS["red"] if m>=7 and m<=10 else COLORS["blue"] if m<=6 else COLORS["yellow"] for m in monthly["month"]],
                marker_line_width=0))
            fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["p75"],
                name="P75", line=dict(color=COLORS["orange"], width=1.5, dash="dot"), mode="lines+markers", marker_size=4))
            fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["max"],
                name="Máximo", line=dict(color=COLORS["red"], width=1), mode="markers", marker_size=5, marker_symbol="x"))
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        with col2:
            st.markdown("#### Distribuição PM2.5 — histograma")
            fig = px.histogram(df[df["pm25"]<100], x="pm25", nbins=40,
                               color="season", color_discrete_map=SEASON_COLORS,
                               labels={"pm25":"PM2.5 µg/m³","season":"Estação"})
            fig.add_vline(x=15, line_dash="dash", line_color=COLORS["orange"],
                          annotation_text="OMS 15µg/m³", annotation_position="top right",
                          annotation_font_size=9)
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Padrão horário por estação")

            def create_pm25_polar_plot(df, season_colors, colors):
                hs = df.groupby(["hour","season"])["pm25"].mean().reset_index()
                fig = px.line_polar(
                    hs, 
                    r="pm25", 
                    theta="hour", 
                    color="season",
                    line_close=True,
                    color_discrete_map=season_colors,
                    template="plotly_dark",
                    start_angle=90,
                    direction="clockwise"
                )
                fig.update_traces(line_width=2.5)
                fig.update_layout(
                    polar=dict(
                        bgcolor=colors["card"],
                        radialaxis=dict(
                            visible=True, 
                            range=[0, hs["pm25"].max() + 5],
                            gridcolor=colors["txt_disabled"],
                            tickfont_color=colors["txt2"]
                        ),
                        angularaxis=dict(
                            type="linear",
                            period=24,
                            tickvals=list(range(0, 24, 2)),
                            ticktext=[f"{h}h" for h in range(0, 24, 2)],
                            direction="clockwise",
                            gridcolor=colors["txt_disabled"],
                            tickfont_color=colors["txt2"],
                        )
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                return fig

            fig = create_pm25_polar_plot(df, SEASON_COLORS, COLORS)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("#### Heatmap PM2.5 — hora × mês")
            pivot = df.groupby(["hour","month"])["pm25"].mean().reset_index()
            pivot["month_name"] = pivot["month"].apply(lambda m: MONTH_LABELS[m-1])
            fig = px.density_heatmap(pivot, x="month_name", y="hour", z="pm25",
                                      color_continuous_scale=["#1A4A7A","#1A6B3C","#B8860B","#C8401A","#7A1A05"],
                                      labels={"month_name":"Mês","hour":"Hora","pm25":"PM2.5 µg/m³"})
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### Estatísticas descritivas — todos os parâmetros")
        cols_stat = ["pm25","no2","co","o3","so2","rain"]
        stats = df[cols_stat].describe().T.round(2)
        stats.index = ["PM2.5 µg/m³","NO2 µg/m³","CO ppm","O3 µg/m³","SO2 µg/m³","Chuva mm"]
        st.dataframe(stats, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: EDA IQAIR
# ══════════════════════════════════════════════════════════════════════════════════
elif "IQAir" in page:
    st.markdown('<div class="sec-tag">03 — IQAir · 4 sensores · Mar 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">EDA — IQAir BRASÍLIA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">36.301 registros de quasi-tempo real. 4 sensores distribuídos em Brasília.</div>', unsafe_allow_html=True)

    if iq is None:
        no_data("IQAir CSV")
    else:
        sel_sensors = st.multiselect("Sensores", iq["sensor_location"].unique().tolist(),
                                     default=iq["sensor_location"].unique().tolist())
        df = iq[iq["sensor_location"].isin(sel_sensors)].copy()

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Registros", f"{len(df):,}")
        with c2: st.metric("AQI médio", f"{df['aqi'].mean():.1f}")
        with c3: st.metric("AQI máximo", f"{df['aqi'].max()}", delta_color="inverse", delta="pico")
        with c4: st.metric("PM2.5 médio", f"{df['pm25'].mean():.2f}")

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### AQI por sensor — box plot")
            fig = px.box(df, x="sensor_location", y="aqi", color="sensor_location",
                         labels={"sensor_location":"Sensor","aqi":"AQI"},
                         color_discrete_sequence=[COLORS["green"],COLORS["blue"],COLORS["yellow"],COLORS["red"]])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        with col2:
            st.markdown("#### AQI diário — todos os sensores")
            daily = df.groupby(["date","sensor_location"])["aqi"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            fig = px.line(daily, x="date", y="aqi", color="sensor_location",
                          labels={"date":"Data","aqi":"AQI","sensor_location":"Sensor"},
                          color_discrete_sequence=[COLORS["blue"],COLORS["green"],COLORS["yellow"],COLORS["red"]])
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Padrão horário AQI por sensor")
            hourly_s = df.groupby(["hour","sensor_location"])["aqi"].mean().reset_index()
            fig = px.line(hourly_s, x="hour", y="aqi", color="sensor_location",
                          labels={"hour":"Hora","aqi":"AQI médio","sensor_location":"Sensor"},
                          color_discrete_sequence=[COLORS["blue"],COLORS["green"],COLORS["yellow"],COLORS["red"]])
            fig.update_traces(line_width=2)
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        with col4:
            st.markdown("#### AQI × Umidade relativa (IQAir)")
            daily2 = df.groupby("date").agg(aqi=("aqi","mean"), rh=("humidity","mean"), temp=("temperature","mean")).reset_index()
            fig = px.scatter(daily2, x="rh", y="aqi", size="aqi",
                             color="temp", color_continuous_scale="RdYlGn_r",
                             labels={"rh":"Umidade %","aqi":"AQI médio","temp":"Temperatura °C"},
                             hover_data={"date":True})
            st.plotly_chart(fig_layout(fig, 300), use_container_width=True)

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 Estatísticas por sensor")
        sensor_stats = df.groupby("sensor_location").agg(
            Registros=("aqi","count"),
            AQI_medio=("aqi","mean"),
            AQI_max=("aqi","max"),
            AQI_std=("aqi","std"),
            PM25_medio=("pm25","mean"),
            Temp_media=("temperature","mean"),
            Umidade_media=("humidity","mean"),
        ).round(2).reset_index()
        sensor_stats.columns = ["Sensor","Registros","AQI Médio","AQI Máx","AQI Desvio","PM2.5 Médio","Temp Média","Umidade Média"]
        st.dataframe(sensor_stats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: CRUZAMENTOS
# ══════════════════════════════════════════════════════════════════════════════════
elif "Cruzamentos" in page:
    st.markdown('<div class="sec-tag">04 — Cruzamento dos datasets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">CRUZAMENTOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Análise de correlações entre variáveis INMET e IQAir. Cada cruzamento valida uma hipótese do modelo preditivo.</div>', unsafe_allow_html=True)

    if inmet is None or iq is None:
        no_data("INMET + IQAir (ambos necessários)")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Matriz de correlação — INMET")
            corr_cols = ["pm25","no2","co","o3","rain"]
            corr_labels = ["PM2.5","NO2","CO","O3","Chuva"]
            cm = inmet[corr_cols].corr().values
            fig = go.Figure(go.Heatmap(
                z=cm, x=corr_labels, y=corr_labels,
                text=[[f"{v:.3f}" for v in row] for row in cm],
                texttemplate="%{text}", textfont_size=10,
                colorscale=[[0,"#C8401A"],[0.5,"#1E2126"],[1,"#1A4A7A"]],
                zmin=-1, zmax=1, colorbar=dict(thickness=10, tickfont_size=9)
            ))
            st.plotly_chart(fig_layout(fig, 320), use_container_width=True)

        with col2:
            st.markdown("#### Scatter PM2.5 × NO2 por estação")
            sample = inmet[inmet["pm25"]<100].sample(min(600, len(inmet)), random_state=42)
            fig = px.scatter(sample, x="no2", y="pm25", color="season",
                             color_discrete_map=SEASON_COLORS, opacity=0.6,
                             trendline="ols",
                             labels={"no2":"NO2 µg/m³","pm25":"PM2.5 µg/m³","season":"Estação"},
                             hover_data={"month":True,"hour":True})
            st.plotly_chart(fig_layout(fig, 320), use_container_width=True)

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Efeito da chuva sobre PM2.5 — INMET 2025")
            daily_inmet = inmet.groupby("date").agg(pm25=("pm25","mean"), rain=("rain","sum")).reset_index()
            bins   = [-1, 0, 5, 20, 10000]
            labels = ["0mm (seco)","1–5mm (leve)","5–20mm (mod.)","≥20mm (forte)"]
            daily_inmet["rain_cat"] = pd.cut(daily_inmet["rain"], bins=bins, labels=labels)
            rain_eff = daily_inmet.groupby("rain_cat", observed=True)["pm25"].mean().reset_index()
            clrs = [COLORS["red"], COLORS["yellow"], COLORS["teal"], COLORS["blue"]]
            fig = px.bar(rain_eff, x="rain_cat", y="pm25", color="rain_cat",
                         color_discrete_sequence=clrs,
                         labels={"rain_cat":"Chuva acumulada diária","pm25":"PM2.5 médio µg/m³"})
            fig.update_layout(showlegend=False)
            fig.add_annotation(x=2, y=rain_eff["pm25"].max()*0.6,
                text="−34% vs dia seco", showarrow=False,
                font=dict(size=11, color=COLORS["teal"]))
            st.plotly_chart(fig_layout(fig, 280), use_container_width=True)

        with col4:
            st.markdown("#### Comparação padrão horário: INMET vs IQAir")
            iq_h   = iq.groupby("hour")["aqi"].mean().reset_index()
            inm_h  = inmet.groupby("hour")["pm25"].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=inm_h["hour"], y=inm_h["pm25"],
                name="INMET PM2.5 µg/m³", line=dict(color=COLORS["red"], width=2.5),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.07)"), secondary_y=False)
            fig.add_trace(go.Scatter(x=iq_h["hour"], y=iq_h["aqi"],
                name="IQAir AQI", line=dict(color=COLORS["blue"], width=2, dash="dot")), secondary_y=True)
            fig.update_yaxes(title_text="PM2.5 µg/m³", secondary_y=False, title_font_color=COLORS["red"])
            fig.update_yaxes(title_text="AQI", secondary_y=True, title_font_color=COLORS["blue"])
            fig.update_xaxes(title_text="Hora do dia")
            st.plotly_chart(fig_layout(fig, 280), use_container_width=True)

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### Temperatura × AQI — IQAir (dia a dia, mar 2026)")
        daily_iq = iq.groupby("date").agg(aqi=("aqi","mean"), temp=("temperature","mean"),
                                            rh=("humidity","mean"), pm25=("pm25","mean")).reset_index()
        fig = px.scatter(daily_iq, x="temp", y="aqi", size="pm25", color="rh",
                         color_continuous_scale="Blues",
                         labels={"temp":"Temperatura °C","aqi":"AQI médio","rh":"Umidade %","pm25":"PM2.5"},
                         trendline="ols")
        fig.update_layout(coloraxis_colorbar=dict(thickness=10))
        st.plotly_chart(fig_layout(fig, 320), use_container_width=True)
        r = daily_iq[["temp","aqi"]].corr().iloc[0,1]
        st.info(f"📊 Correlação Temperatura × AQI: **r = {r:.3f}** — temperatura mais baixa associada a AQI menor (ar úmido = melhor dispersão)")


# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: MODELO
# ══════════════════════════════════════════════════════════════════════════════════
elif "Modelo" in page:
    st.markdown('<div class="sec-tag">05 — Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">MODELO PREDITIVO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">XGBoost treinado com dados INMET 2025. Previsão de PM2.5 e cálculo do Índice de Saída 0–100 para o motociclista.</div>', unsafe_allow_html=True)

    if inmet is None:
        no_data("INMET 2025 XLSX")
    else:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.preprocessing import LabelEncoder
        import pickle

        @st.cache_resource
        def train_model(df_raw):
            df = df_raw.copy()
            df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
            df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
            df["month_sin"]= np.sin(2*np.pi*df["month"]/12)
            df["month_cos"]= np.cos(2*np.pi*df["month"]/12)
            df["is_dry"] = (df["month"].between(7,10)).astype(int)
            df["pm25_lag1"] = df["pm25"].shift(1).fillna(df["pm25"].mean())
            df["pm25_lag3"] = df["pm25"].shift(3).fillna(df["pm25"].mean())
            df["pm25_roll3"] = df["pm25"].rolling(3, min_periods=1).mean()
            df["rain_acc6"] = df["rain"].rolling(6, min_periods=1).sum()

            features = ["hour_sin","hour_cos","month_sin","month_cos","is_dry",
                        "pm25_lag1","pm25_lag3","pm25_roll3","rain_acc6","no2","co"]
            target = "pm25"
            df_ml = df[features + [target]].dropna()
            X, y = df_ml[features], df_ml[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     random_state=42, verbosity=0)
            except ImportError:
                model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            fi   = pd.Series(model.feature_importances_, index=features, name="importance").sort_values(ascending=False)
            return model, mae, r2, fi, y_test.values[:100], y_pred[:100], features

        with st.spinner("🤖 Treinando modelo XGBoost..."):
            model, mae, r2, fi, y_true, y_pred, features = train_model(inmet)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("MAE",         f"{mae:.2f} µg/m³")
        with c2: st.metric("R²",          f"{r2:.3f}", delta="variância explicada")
        with c3: st.metric("Features",    f"{len(features)}")
        with c4: st.metric("Amostras treino", f"{int(len(inmet)*0.8):,}")
        with st.expander("📋 Detalhes do modelo"):
            st.write(f"**MAE:** {mae:.2f} µg/m³")
            st.write(f"**R²:** {r2:.3f}")
            st.write(f"**Features:** {len(features)}")

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Importância das features")
            fig = px.bar(fi.reset_index(), x="importance", y="index", orientation="h",
                         color="importance", color_continuous_scale=["#1A4A7A","#FF5722"],
                         labels={"importance":"Importância","index":"Feature"})
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig_layout(fig, 320), use_container_width=True)

        with col2:
            st.markdown("#### Real vs Previsto (amostra de 100 pontos)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_true))), y=y_true,
                name="Real", line=dict(color=COLORS["teal"], width=1.5)))
            fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred,
                name="Previsto", line=dict(color=COLORS["accent"], width=1.5, dash="dot")))
            fig.update_layout(xaxis_title="Amostras", yaxis_title="PM2.5 µg/m³")
            st.plotly_chart(fig_layout(fig, 320), use_container_width=True)

        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### 🎮 Simulador — Índice de Saída do Motociclista")
        st.caption("Ajuste os parâmetros para calcular o MotoAR Score em tempo real")

        s1, s2, s3 = st.columns(3)
        with s1:
            sim_hour  = st.slider("Hora do dia", 0, 23, 8)
            sim_month = st.selectbox("Mês", range(1,13), index=6,
                                     format_func=lambda m: MONTH_LABELS[m-1])
        with s2:
            sim_pm25_lag = st.slider("PM2.5 última hora (µg/m³)", 0.0, 80.0, 10.0, 0.5)
            sim_rain     = st.slider("Chuva acumulada 6h (mm)", 0.0, 50.0, 0.0, 0.5)
        with s3:
            sim_no2 = st.slider("NO2 (µg/m³)", 0.0, 80.0, 10.0, 0.5)
            sim_co  = st.slider("CO (ppm)", 0.0, 3.0, 0.5, 0.05)

        is_dry = 1 if 7 <= sim_month <= 10 else 0
        X_sim = pd.DataFrame([{
            "hour_sin":   np.sin(2*np.pi*sim_hour/24),
            "hour_cos":   np.cos(2*np.pi*sim_hour/24),
            "month_sin":  np.sin(2*np.pi*sim_month/12),
            "month_cos":  np.cos(2*np.pi*sim_month/12),
            "is_dry":     is_dry,
            "pm25_lag1":  sim_pm25_lag,
            "pm25_lag3":  sim_pm25_lag,
            "pm25_roll3": sim_pm25_lag,
            "rain_acc6":  sim_rain,
            "no2":        sim_no2,
            "co":         sim_co,
        }])
        pm25_pred = float(model.predict(X_sim)[0])
        pm25_pred = max(0, pm25_pred)

        # Índice de saída 0–100 (quanto menor PM2.5, melhor)
        score = max(0, min(100, round(100 - (pm25_pred / 50) * 80 - (is_dry * 10))))

        if score >= 75:
            score_color, score_label = COLORS["green"],  "Ótimo para sair"
        elif score >= 55:
            score_color, score_label = COLORS["yellow"], "Favorável"
        elif score >= 35:
            score_color, score_label = COLORS["orange"], "Use filtro de ar"
        else:
            score_color, score_label = COLORS["red"],    "Evite ou use EPI completo"

        sc1, sc2, sc3 = st.columns([1,1,2])
        with sc1:
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid {score_color};border-radius:14px;padding:1.5rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8A9099;text-transform:uppercase;margin-bottom:8px;">PM2.5 previsto</div>
                <div style="font-family:'Barlow Condensed';font-size:48px;font-weight:900;color:{score_color};line-height:1;">{pm25_pred:.1f}</div>
                <div style="font-size:13px;color:#8A9099;">µg/m³</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div style="background:#1E2126;border:0.5px solid {score_color};border-radius:14px;padding:1.5rem;text-align:center;">
                <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8A9099;text-transform:uppercase;margin-bottom:8px;">MotoAR Score</div>
                <div style="font-family:'Barlow Condensed';font-size:48px;font-weight:900;color:{score_color};line-height:1;">{score}</div>
                <div style="font-size:13px;color:{score_color};font-weight:600;">{score_label}</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            gear = []
            if pm25_pred > 12: gear.append("🪖 Capacete com filtro de ar — PM2.5 elevado")
            if pm25_pred > 25: gear.append("😷 Máscara N95 recomendada — nível preocupante")
            if sim_no2 > 40:   gear.append("🕶️ Viseira fechada — NO2 elevado")
            if sim_rain > 5:   gear.append("🌂 Capa de chuva — precipitação recente")
            if sim_month in [6,7,8]: gear.append("🧥 Jaqueta com forro — mês mais frio")
            if not gear:       gear = ["✅ Condições favoráveis — equipamento padrão suficiente"]
            st.markdown("**Gear recomendado:**")
            for g in gear:
                st.markdown(f"- {g}")

# GUARDA MODELO TREINADO EM PASTA QUE SERA CRIADA PARA ISSO COM O NOME "model" E O ARQUIVO "xgb_model.pkl". USAR PICKLE PARA ISSO. INCLUIR BOTÃO PARA DOWNLOAD DO MODELO TREINADO.
        with st.expander("💾 Baixar modelo treinado"):
            import io
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)
            st.download_button(
                label="Download do modelo XGBoost",
                data=buffer,
                file_name="xgb_model.pkl",
                mime="application/octet-stream" )
 
            

# ══════════════════════════════════════════════════════════════════════════════════
# PÁGINA: GEAR
# ══════════════════════════════════════════════════════════════════════════════════
elif "Gear" in page:
    st.markdown('<div class="sec-tag">06 — Recomendação de equipamento</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">GUIA DE EQUIPAMENTO — MOTOAR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Lógica de recomendação baseada nos dados reais INMET 2025 + IQAir Mar 2026. Thresholds calibrados para Brasília-DF.</div>', unsafe_allow_html=True)

    gear_data = [
        ["🪖","Capacete c/ filtro de ar","PM2.5 > 12 µg/m³ (meta OMS 24h)","INMET + IQAir","Jul–Out 65% das horas","Alta","b-red"],
        ["😷","Máscara N95","PM2.5 > 25 µg/m³ (nível OMS ruim)","INMET","Set–Out: picos extremos","Alta","b-red"],
        ["🧥","Jaqueta pesada / forro","Temperatura < 18°C","IQAir + Open-Meteo","Jun–Ago: manhãs frias","Média","b-yellow"],
        ["🧥","Jaqueta ventilada","Temperatura > 30°C","IQAir","Set–Nov: dias quentes secos","Média","b-orange"],
        ["💧","Hidratação extra (1L+)","Temp > 30°C E Umidade < 40%","IQAir + INMET","Set–Nov: 30% dos dias secos","Média","b-yellow"],
        ["🌂","Capa de chuva obrigatória","Precipitação prevista > 2mm/h","Open-Meteo forecast","Nov–Mar: ~3×/semana","Alta","b-red"],
        ["🕶️","Óculos / viseira fechada","Vento > 30 km/h OU PM2.5 > 25","INMET (vento) + PM2.5","Variável por rota","Média","b-yellow"],
        ["🧤","Luvas de frio","Temperatura < 15°C","IQAir + previsão","Jun–Jul: manhãs extremas","Baixa","b-blue"],
        ["🌡️","Monitor pessoal de qualidade","PM2.5 local desconhecido","—","Regiões sem sensor","Baixa","b-blue"],
    ]
    SEASON_COLORS = {"🌧️ Chuva": COLORS["blue"], "🔥 Seca": COLORS["red"], "🍂 Transição": COLORS["yellow"]}
    
    
    def generate_full_report_pdf():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        import matplotlib.ticker as mticker

        BG      = "#0D0F12"
        CARD    = "#1E2126"
        TXT     = "#F0F2F5"
        TXT2    = "#8A9099"
        ACCENT  = "#FF5722"
        C_RED   = "#E74C3C"
        C_BLUE  = "#3498DB"
        C_GREEN = "#2ECC71"
        C_YEL   = "#F1C40F"
        C_TEAL  = "#1ABC9C"
        C_ORAN  = "#FF8C00"
        MONTH_L = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": CARD,
            "text.color": TXT, "axes.labelcolor": TXT2,
            "xtick.color": TXT2, "ytick.color": TXT2,
            "axes.edgecolor": "#333840", "grid.color": "#333840",
            "grid.linewidth": 0.5, "axes.titlesize": 10,
            "axes.titlecolor": TXT, "axes.titleweight": "bold",
            "font.family": "monospace",
        })

        def fig_to_image(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            buf.seek(0)
            plt.close(fig)
            return buf

        story_elements = []
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=36, rightMargin=36,
                                topMargin=36, bottomMargin=36)
        rl_styles = getSampleStyleSheet()

        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch

        style_cover = ParagraphStyle("cover", parent=rl_styles["Title"],
                                     fontSize=28, textColor=colors.HexColor("#FF5722"),
                                     spaceAfter=6)
        style_sub   = ParagraphStyle("sub", parent=rl_styles["Normal"],
                                     fontSize=10, textColor=colors.HexColor("#8A9099"),
                                     spaceAfter=18)
        style_h1    = ParagraphStyle("h1", parent=rl_styles["Heading1"],
                                     fontSize=15, textColor=colors.HexColor("#FF5722"),
                                     spaceBefore=18, spaceAfter=6,
                                     borderPad=2)
        style_h2    = ParagraphStyle("h2", parent=rl_styles["Heading2"],
                                     fontSize=11, textColor=colors.HexColor("#F0F2F5"),
                                     spaceBefore=10, spaceAfter=4)
        style_body  = ParagraphStyle("body", parent=rl_styles["Normal"],
                                     fontSize=9, textColor=colors.HexColor("#C0C4CC"),
                                     leading=14, spaceAfter=6)
        style_tag   = ParagraphStyle("tag", parent=rl_styles["Normal"],
                                     fontSize=8, textColor=colors.HexColor("#FF5722"),
                                     spaceAfter=2)

        def tbl_style(header_bg="#1E2126"):
            return TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#FF5722")),
                ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
                ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#1E2126"),colors.HexColor("#16191C")]),
                ("TEXTCOLOR",   (0,1), (-1,-1), colors.HexColor("#C0C4CC")),
                ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#333840")),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING",(0,0), (-1,-1), 6),
                ("TOPPADDING",  (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                ("ALIGN",       (0,0), (-1,-1), "LEFT"),
                ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ])

        def add_chart(fig, width=6.8*inch, height=2.6*inch):
            img_buf = fig_to_image(fig)
            story_elements.append(Image(img_buf, width=width, height=height))
            story_elements.append(Spacer(1, 6))

        # ── CAPA ──────────────────────────────────────────────────────────────────
        story_elements.append(Spacer(1, 40))
        story_elements.append(Paragraph("MotoAR", style_cover))
        story_elements.append(Paragraph("Análise de Qualidade do Ar para Motociclistas", style_sub))
        story_elements.append(Paragraph("Brasília-DF · INMET 2025 + IQAir Mar 2026 · XGBoost", style_sub))
        story_elements.append(Spacer(1, 12))

        metrics_data = [
            ["Registros Totais", "Parâmetros", "Sensores", "Cobertura", "Acurácia Modelo"],
            ["45.061", "18", "6", "14 meses", "~82%"],
        ]
        t = Table(metrics_data, colWidths=[1.36*inch]*5)
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(Spacer(1, 6))

        from reportlab.platypus import PageBreak
        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 1 — VISÃO GERAL
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("00 — DASHBOARD", style_tag))
        story_elements.append(Paragraph("Visão Geral do Projeto", style_h1))
        story_elements.append(Paragraph(
            "App de qualidade do ar para motociclistas de Brasília. Combina dados INMET "
            "(estação fixa horária) com IQAir (4 sensores quasi-tempo real) para recomendar "
            "equipamentos e horários de saída.", style_body))

        # Gráfico 1a: Sazonalidade PM2.5
        monthly_pm = [5.1,7.13,9.49,6.07,4.2,5.84,17.77,18.3,22.25,18.82,12.12,9.44]
        bar_colors = [C_BLUE if i<6 else C_RED if i<10 else C_YEL for i in range(12)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))

        ax = axes[0]
        bars = ax.bar(MONTH_L, monthly_pm, color=bar_colors, edgecolor="none", linewidth=0)
        ax.axhline(15, color=C_ORAN, linewidth=1.2, linestyle="--", label="OMS 15 µg/m³")
        for bar, v in zip(bars, monthly_pm):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7, color=TXT2)
        ax.set_title("Sazonalidade PM2.5 — CRAS Fercal 2025")
        ax.set_ylabel("µg/m³", color=TXT2)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor(CARD)

        # Gráfico 1b: Padrão horário por estação
        hourly_seca  = [28.09,24.93,21.19,19.01,17.46,16.61,16.33,17.27,21.22,23.51,18.0,9.91,8.24,7.52,7.97,7.06,8.91,12.58,16.86,26.69,34.0,33.43,31.91,30.72]
        hourly_chuva = [7.23,5.77,4.81,4.12,4.11,3.82,4.11,6.32,8.35,8.21,5.64,3.83,3.66,3.9,4.21,4.99,5.05,5.91,6.55,9.28,11.53,10.62,8.59,8.2]
        hourly_trans = [8.83,7.66,6.92,7.8,8.45,9.12,11.25,15.8,13.22,9.04,6.72,8.09,6.07,4.54,6.4,6.27,9.33,10.74,15.22,21.58,20.31,16.98,14.74,11.97]
        horas = list(range(24))
        ax2 = axes[1]
        ax2.plot(horas, hourly_seca,  color=C_RED,  linewidth=2.2, label="Seca (Jul-Out)")
        ax2.fill_between(horas, hourly_seca,  alpha=0.08, color=C_RED)
        ax2.plot(horas, hourly_trans, color=C_YEL,  linewidth=1.5, linestyle="--", label="Transicao")
        ax2.plot(horas, hourly_chuva, color=C_BLUE, linewidth=2,   label="Chuva (Jan-Jun)")
        ax2.fill_between(horas, hourly_chuva, alpha=0.07, color=C_BLUE)
        ax2.axhline(15, color=C_ORAN, linewidth=1, linestyle=":", alpha=0.7)
        ax2.set_title("Padrao horario — PM2.5 por estacao")
        ax2.set_xlabel("Hora", color=TXT2)
        ax2.set_ylabel("PM2.5 µg/m³", color=TXT2)
        ax2.legend(fontsize=7)
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_facecolor(CARD)
        fig.tight_layout(pad=1.2)
        add_chart(fig, height=2.8*inch)

        # Achados principais — tabela
        story_elements.append(Paragraph("Achados Principais", style_h2))
        findings_data = [
            ["Achado", "Detalhe"],
            ["Amplitude sazonal 5.3x", "PM2.5 vai de 4,2 µg/m3 (mai) a 22,3 µg/m3 (set). Queimadas dominam."],
            ["Dois picos diarios",     "7-9h (trafego) e 19-21h (inversao termica). Confirmado INMET e IQAir."],
            ["Chuva reduz PM2.5 34%",  "Dias >5mm chuva: PM2.5 medio 7,9 vs 12,8 nos dias secos."],
            ["70% diferenca espacial", "UnB Gama tem AQI 70% maior que Finatec. Rota importa tanto quanto hora."],
            ["NO2 correlaciona PM2.5", "r=0.47. NO2 como proxy de combustao (trafego + queimadas)."],
            ["65% horas exigem filtro","Estacao seca: 65% das horas noturnas > limiar OMS de 15 µg/m3."],
        ]
        t = Table(findings_data, colWidths=[2.1*inch, 4.7*inch])
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 2 — LCA
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("01 — CICLO DE VIDA", style_tag))
        story_elements.append(Paragraph("LCA — MotoAR", style_h1))
        story_elements.append(Paragraph(
            "Mapeamento completo do ciclo de vida dos dados: da coleta bruta ao MotoAR Score "
            "entregue ao motociclista.", style_body))

        # Pipeline visual
        pipeline_data = [
            ["Etapa", "Nome", "Acoes principais"],
            ["01", "Ingestao",    "IQAir API · INMET estacoes automaticas · Open-Meteo forecast · Polling 5min"],
            ["02", "Limpeza",     "Filtro status Ok · Clipping outliers · Imputacao nulos · Normalizacao unidades"],
            ["03", "Feature Eng.","Hora sin/cos · Flag seca/chuva · Rolling avg 3h/24h · Chuva acumulada 6h"],
            ["04", "Modelo",      "XGBoost · Indice Saida 0-100 · Previsao PM2.5 +6h · Classificacao risco"],
            ["05", "Entrega",     "Push notification · Tela Agora MotoAR · Alerta queimadas Jul-Out · Gear"],
        ]
        t = Table(pipeline_data, colWidths=[0.5*inch, 1.2*inch, 5.1*inch])
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(Spacer(1, 10))

        # Inventário de dados
        story_elements.append(Paragraph("Inventario de Dados", style_h2))
        inventory_data = [
            ["Fonte", "Cobertura", "Registros", "Parametros", "Freq.", "Status"],
            ["INMET — CRAS Fercal",     "Jan-Dez 2025", "8.760",  "14 params", "1h",    "Ativo"],
            ["INMET — Estacao Escola",  "Jan-Dez 2025", "8.760",  "8 params",  "1h",    "Ativo"],
            ["IQAir — Brasilia centro", "Mar 2026",     "9.853",  "AQI+PM2.5", "~1min", "Live"],
            ["IQAir — Escola 115 Norte","Mar 2026",     "8.766",  "AQI+PM2.5", "~1min", "Live"],
            ["IQAir — UnB Gama",        "Mar 2026",     "9.053",  "AQI+PM2.5", "~1min", "Live"],
            ["IQAir — Finatec",         "Mar 2026",     "8.629",  "AQI+PM2.5", "~1min", "Live"],
            ["Open-Meteo (planejado)",  "Futuro",       "—",      "Temp+Precip","1h",   "Integrar"],
            ["INMET 2024 (planejado)",  "Jan-Dez 2024", "8.784",  "Mesmos",    "1h",    "Pendente"],
        ]
        cws = [2.0*inch, 1.0*inch, 0.75*inch, 0.9*inch, 0.55*inch, 0.65*inch]
        t = Table(inventory_data, colWidths=cws)
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(Spacer(1, 10))

        # Limitações
        story_elements.append(Paragraph("Limitacoes Identificadas", style_h2))
        limits_data = [
            ["Limitacao", "Impacto"],
            ["INMET — sensor temperatura invalido",  "Valores negativos extremos filtrados"],
            ["IQAir — apenas mar/2026",              "Serie curta; ideal >= 6 meses"],
            ["INMET 2024 — nao recebido ainda",      "Reduz capacidade de treino do modelo"],
            ["Cobertura espacial limitada",           "6 sensores para toda Brasilia-DF"],
            ["PM2.5 IQAir em escala AQI",            "Nao diretamente comparavel com INMET µg/m3"],
        ]
        t = Table(limits_data, colWidths=[3.0*inch, 3.8*inch])
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 3 — EDA INMET
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("02 — ANALISE EXPLORATORIA", style_tag))
        story_elements.append(Paragraph("EDA — INMET 2025", style_h1))
        story_elements.append(Paragraph(
            "Estacao CRAS Fercal · Serie horaria completa Jan-Dez 2025 · 14 parametros ambientais.", style_body))

        if inmet is not None:
            # Métricas INMET
            m_data = [
                ["Registros validos PM2.5", "PM2.5 medio", "PM2.5 maximo", "Mediana (P50)", "Desvio padrao"],
                [f"{len(inmet):,}", f"{inmet['pm25'].mean():.1f} µg/m3",
                 f"{inmet['pm25'].max():.1f} µg/m3", f"{inmet['pm25'].median():.1f} µg/m3",
                 f"{inmet['pm25'].std():.1f} µg/m3"],
            ]
            t = Table(m_data, colWidths=[1.36*inch]*5)
            t.setStyle(tbl_style())
            story_elements.append(t)
            story_elements.append(Spacer(1, 8))

            # Gráficos EDA INMET (2x2)
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))

            # 1) PM2.5 mensal — média, P75, máx
            monthly = inmet.groupby("month")["pm25"].agg(["mean","median",
                lambda x: x.quantile(.75), "max"]).reset_index()
            monthly.columns = ["month","mean","median","p75","max"]
            ax = axes[0][0]
            bar_c = [C_RED if 7<=m<=10 else C_BLUE if m<=6 else C_YEL for m in monthly["month"]]
            ax.bar(MONTH_L[:len(monthly)], monthly["mean"], color=bar_c, edgecolor="none", label="Media")
            ax.plot(MONTH_L[:len(monthly)], monthly["p75"],  color=C_ORAN, linewidth=1.5,
                    linestyle="--", marker="o", markersize=3, label="P75")
            ax.plot(MONTH_L[:len(monthly)], monthly["max"],  color=C_RED, linewidth=0,
                    marker="x", markersize=5, label="Maximo")
            ax.legend(fontsize=7)
            ax.set_title("PM2.5 mensal — media, P75, maximo")
            ax.set_ylabel("µg/m³", color=TXT2)
            ax.grid(axis="y", alpha=0.3)

            # 2) Histograma PM2.5 por estação
            ax = axes[0][1]
            s_colors = {"🌧️ Chuva": C_BLUE, "🔥 Seca": C_RED, "🍂 Transição": C_YEL}
            for season, sc in s_colors.items():
                sub = inmet[(inmet["season"]==season) & (inmet["pm25"]<100)]["pm25"]
                if len(sub):
                    ax.hist(sub, bins=30, alpha=0.65, color=sc, label=season.split()[0]+" "+season.split()[1], edgecolor="none")
            ax.axvline(15, color=C_ORAN, linewidth=1.2, linestyle="--", label="OMS 15")
            ax.set_title("Distribuicao PM2.5 por estacao")
            ax.set_xlabel("PM2.5 µg/m³", color=TXT2)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

            # 3) Padrão horário por estação
            ax = axes[1][0]
            hs = inmet.groupby(["hour","season"])["pm25"].mean().reset_index()
            for season, sc in s_colors.items():
                sub = hs[hs["season"]==season]
                if len(sub):
                    lbl = season.split()[0]+" "+season.split()[1]
                    ax.plot(sub["hour"], sub["pm25"], color=sc, linewidth=2.2, label=lbl)
            ax.axhline(15, color=C_GREEN, linewidth=1, linestyle="--", alpha=0.6, label="OMS 24h")
            ax.set_title("Padrao horario por estacao")
            ax.set_xlabel("Hora", color=TXT2)
            ax.set_ylabel("PM2.5 µg/m³", color=TXT2)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

            # 4) Heatmap PM2.5 hora × mês
            ax = axes[1][1]
            pivot = inmet.groupby(["hour","month"])["pm25"].mean().unstack(fill_value=0)
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                           extent=[-0.5, 11.5, 23.5, -0.5], origin="upper")
            ax.set_xticks(range(12))
            ax.set_xticklabels(MONTH_L, fontsize=7)
            ax.set_ylabel("Hora", color=TXT2)
            ax.set_title("Heatmap PM2.5 — hora x mes")
            plt.colorbar(im, ax=ax, shrink=0.8, label="µg/m³")

            for a in axes.flat:
                a.set_facecolor(CARD)
            fig.tight_layout(pad=1.4)
            add_chart(fig, height=4.6*inch)

            # Estatísticas descritivas
            story_elements.append(Paragraph("Estatisticas Descritivas — Poluentes", style_h2))
            cols_stat = ["pm25","no2","co","o3","so2","rain"]
            labels_stat = ["PM2.5 µg/m3","NO2 µg/m3","CO ppm","O3 µg/m3","SO2 µg/m3","Chuva mm"]
            stats = inmet[cols_stat].describe().T.round(2)
            stats_rows = [["Parametro","Count","Mean","Std","Min","P25","P50","P75","Max"]]
            for lbl, row in zip(labels_stat, stats.itertuples()):
                stats_rows.append([lbl, str(int(row.count)), f"{row.mean:.2f}",
                                   f"{row.std:.2f}", f"{row.min:.2f}",
                                   f"{getattr(row,'25%'):.2f}", f"{getattr(row,'50%'):.2f}",
                                   f"{getattr(row,'75%'):.2f}", f"{row.max:.2f}"])
            cws2 = [1.3*inch]+[0.72*inch]*8
            t = Table(stats_rows, colWidths=cws2)
            t.setStyle(tbl_style())
            story_elements.append(t)
        else:
            story_elements.append(Paragraph("⚠ Dados INMET nao carregados — graficos indisponiveis.", style_body))

        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 4 — EDA IQAIR
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("03 — IQAIR · 4 SENSORES · MAR 2026", style_tag))
        story_elements.append(Paragraph("EDA — IQAir Brasilia", style_h1))
        story_elements.append(Paragraph(
            "36.301 registros de quasi-tempo real. 4 sensores distribuidos em Brasilia.", style_body))

        if iq is not None:
            m_data = [
                ["Registros", "AQI medio", "AQI maximo", "PM2.5 medio"],
                [f"{len(iq):,}", f"{iq['aqi'].mean():.1f}", f"{iq['aqi'].max()}", f"{iq['pm25'].mean():.2f}"],
            ]
            t = Table(m_data, colWidths=[1.7*inch]*4)
            t.setStyle(tbl_style())
            story_elements.append(t)
            story_elements.append(Spacer(1, 8))

            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            sensor_colors = [C_GREEN, C_BLUE, C_YEL, C_RED]
            sensors = iq["sensor_location"].unique().tolist()

            # 1) Box plot AQI por sensor
            ax = axes[0][0]
            bp_data = [iq[iq["sensor_location"]==s]["aqi"].dropna().values for s in sensors]
            bp = ax.boxplot(bp_data, patch_artist=True, widths=0.5,
                            medianprops=dict(color=TXT, linewidth=1.5))
            for patch, sc in zip(bp["boxes"], sensor_colors):
                patch.set_facecolor(sc)
                patch.set_alpha(0.7)
            ax.set_xticklabels([s[:15] for s in sensors], fontsize=6, rotation=10)
            ax.set_title("AQI por sensor — box plot")
            ax.set_ylabel("AQI", color=TXT2)
            ax.grid(axis="y", alpha=0.3)

            # 2) AQI diário
            ax = axes[0][1]
            daily = iq.groupby(["date","sensor_location"])["aqi"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            for s, sc in zip(sensors, sensor_colors):
                sub = daily[daily["sensor_location"]==s]
                ax.plot(sub["date"], sub["aqi"], color=sc, linewidth=1.5, label=s[:12])
            ax.set_title("AQI diario — todos os sensores")
            ax.set_xlabel("Data", color=TXT2)
            ax.legend(fontsize=6)
            ax.grid(axis="y", alpha=0.3)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

            # 3) Padrão horário AQI
            ax = axes[1][0]
            hourly_s = iq.groupby(["hour","sensor_location"])["aqi"].mean().reset_index()
            for s, sc in zip(sensors, sensor_colors):
                sub = hourly_s[hourly_s["sensor_location"]==s]
                ax.plot(sub["hour"], sub["aqi"], color=sc, linewidth=2, label=s[:12])
            ax.set_title("Padrao horario AQI por sensor")
            ax.set_xlabel("Hora", color=TXT2)
            ax.set_ylabel("AQI medio", color=TXT2)
            ax.legend(fontsize=6)
            ax.grid(axis="y", alpha=0.3)

            # 4) AQI × Umidade
            ax = axes[1][1]
            daily2 = iq.groupby("date").agg(aqi=("aqi","mean"), rh=("humidity","mean"),
                                             temp=("temperature","mean")).reset_index()
            sc_plot = ax.scatter(daily2["rh"], daily2["aqi"], c=daily2["temp"],
                                  cmap="RdYlGn_r", alpha=0.75, s=40, edgecolors="none")
            plt.colorbar(sc_plot, ax=ax, shrink=0.8, label="Temp °C")
            ax.set_title("AQI x Umidade relativa (IQAir)")
            ax.set_xlabel("Umidade %", color=TXT2)
            ax.set_ylabel("AQI medio", color=TXT2)
            ax.grid(alpha=0.3)

            for a in axes.flat:
                a.set_facecolor(CARD)
            fig.tight_layout(pad=1.4)
            add_chart(fig, height=4.6*inch)

            # Tabela estatísticas por sensor
            story_elements.append(Paragraph("Estatisticas por Sensor", style_h2))
            sensor_stats = iq.groupby("sensor_location").agg(
                Registros=("aqi","count"), AQI_med=("aqi","mean"),
                AQI_max=("aqi","max"), AQI_std=("aqi","std"),
                PM25=("pm25","mean"), Temp=("temperature","mean"),
                Umid=("humidity","mean"),
            ).round(2).reset_index()
            ss_rows = [["Sensor","Registros","AQI Med","AQI Max","AQI Std","PM2.5","Temp","Umid"]]
            for _, row in sensor_stats.iterrows():
                ss_rows.append([str(row["sensor_location"])[:22],
                                str(int(row["Registros"])),
                                f"{row['AQI_med']:.1f}", f"{row['AQI_max']:.0f}",
                                f"{row['AQI_std']:.1f}", f"{row['PM25']:.2f}",
                                f"{row['Temp']:.1f}", f"{row['Umid']:.1f}"])
            cws3 = [2.2*inch]+[0.72*inch]*7
            t = Table(ss_rows, colWidths=cws3)
            t.setStyle(tbl_style())
            story_elements.append(t)
        else:
            story_elements.append(Paragraph("⚠ Dados IQAir nao carregados — graficos indisponiveis.", style_body))

        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 5 — CRUZAMENTOS
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("04 — CRUZAMENTO DOS DATASETS", style_tag))
        story_elements.append(Paragraph("Cruzamentos INMET × IQAir", style_h1))
        story_elements.append(Paragraph(
            "Analise de correlacoes entre variaveis INMET e IQAir. Cada cruzamento valida "
            "uma hipotese do modelo preditivo.", style_body))

        if inmet is not None and iq is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))

            # 1) Matriz de correlação INMET
            ax = axes[0][0]
            corr_cols = ["pm25","no2","co","o3","rain"]
            corr_labels_c = ["PM2.5","NO2","CO","O3","Chuva"]
            cm = inmet[corr_cols].corr().values
            im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xticklabels(corr_labels_c, fontsize=8)
            ax.set_yticklabels(corr_labels_c, fontsize=8)
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", fontsize=8, color=TXT)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Matriz de correlacao — INMET")

            # 2) Scatter PM2.5 × NO2
            ax = axes[0][1]
            for season, sc in {"🌧️ Chuva": C_BLUE, "🔥 Seca": C_RED, "🍂 Transição": C_YEL}.items():
                sub = inmet[(inmet["season"]==season) & (inmet["pm25"]<100)].sample(
                    min(200, len(inmet[inmet["season"]==season])), random_state=42)
                lbl = season.split()[0]+" "+season.split()[1]
                ax.scatter(sub["no2"], sub["pm25"], alpha=0.5, color=sc, s=12, label=lbl, edgecolors="none")
            ax.set_title("Scatter PM2.5 x NO2 por estacao")
            ax.set_xlabel("NO2 µg/m³", color=TXT2)
            ax.set_ylabel("PM2.5 µg/m³", color=TXT2)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # 3) Efeito da chuva sobre PM2.5
            ax = axes[1][0]
            daily_inmet = inmet.groupby("date").agg(pm25=("pm25","mean"), rain=("rain","sum")).reset_index()
            bins_r = [-1, 0, 5, 20, 10000]
            labels_r = ["0mm","1-5mm","5-20mm",">=20mm"]
            daily_inmet["rain_cat"] = pd.cut(daily_inmet["rain"], bins=bins_r, labels=labels_r)
            rain_eff = daily_inmet.groupby("rain_cat", observed=True)["pm25"].mean()
            clrs_r = [C_RED, C_YEL, C_TEAL, C_BLUE]
            ax.bar(rain_eff.index.astype(str), rain_eff.values, color=clrs_r, edgecolor="none")
            for i, v in enumerate(rain_eff.values):
                ax.text(i, v+0.2, f"{v:.1f}", ha="center", fontsize=8, color=TXT2)
            ax.set_title("Efeito da chuva sobre PM2.5 — INMET")
            ax.set_xlabel("Chuva acumulada diaria", color=TXT2)
            ax.set_ylabel("PM2.5 medio µg/m³", color=TXT2)
            ax.grid(axis="y", alpha=0.3)

            # 4) Padrão horário INMET vs IQAir
            ax = axes[1][1]
            iq_h  = iq.groupby("hour")["aqi"].mean()
            inm_h = inmet.groupby("hour")["pm25"].mean()
            ax2b  = ax.twinx()
            ax.plot(inm_h.index, inm_h.values, color=C_RED, linewidth=2.5, label="INMET PM2.5")
            ax.fill_between(inm_h.index, inm_h.values, alpha=0.07, color=C_RED)
            ax2b.plot(iq_h.index, iq_h.values, color=C_BLUE, linewidth=2, linestyle="--", label="IQAir AQI")
            ax.set_ylabel("PM2.5 µg/m³", color=C_RED)
            ax2b.set_ylabel("AQI", color=C_BLUE)
            ax.set_xlabel("Hora do dia", color=TXT2)
            ax.set_title("Padrao horario: INMET vs IQAir")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, fontsize=7)
            ax.grid(axis="y", alpha=0.3)

            for a in axes.flat:
                a.set_facecolor(CARD)
            fig.tight_layout(pad=1.4)
            add_chart(fig, height=4.6*inch)

            # Correlação temperatura × AQI
            daily_iq = iq.groupby("date").agg(aqi=("aqi","mean"), temp=("temperature","mean")).reset_index()
            r = daily_iq[["temp","aqi"]].corr().iloc[0,1]
            story_elements.append(Paragraph(
                f"Correlacao Temperatura x AQI: r = {r:.3f} — temperatura mais baixa "
                "associada a AQI menor (ar umido = melhor dispersao).", style_body))
        else:
            story_elements.append(Paragraph("⚠ Dados INMET e/ou IQAir nao carregados.", style_body))

        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 6 — MODELO PREDITIVO
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("05 — MACHINE LEARNING", style_tag))
        story_elements.append(Paragraph("Modelo Preditivo XGBoost", style_h1))
        story_elements.append(Paragraph(
            "XGBoost treinado com dados INMET 2025. Previsao de PM2.5 e calculo do "
            "Indice de Saida 0-100 para o motociclista.", style_body))

        if inmet is not None:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score

            @st.cache_resource
            def _train_for_pdf(df_raw):
                df = df_raw.copy()
                df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
                df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
                df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
                df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
                df["is_dry"]     = (df["month"].between(7,10)).astype(int)
                df["pm25_lag1"]  = df["pm25"].shift(1).fillna(df["pm25"].mean())
                df["pm25_lag3"]  = df["pm25"].shift(3).fillna(df["pm25"].mean())
                df["pm25_roll3"] = df["pm25"].rolling(3,min_periods=1).mean()
                df["rain_acc6"]  = df["rain"].rolling(6,min_periods=1).sum()
                features = ["hour_sin","hour_cos","month_sin","month_cos","is_dry",
                            "pm25_lag1","pm25_lag3","pm25_roll3","rain_acc6","no2","co"]
                df_ml = df[features+["pm25"]].dropna()
                X, y  = df_ml[features], df_ml["pm25"]
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                try:
                    from xgboost import XGBRegressor
                    mdl = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                       random_state=42, verbosity=0)
                except ImportError:
                    mdl = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
                mdl.fit(X_tr, y_tr)
                yp = mdl.predict(X_te)
                fi = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False)
                return mean_absolute_error(y_te, yp), r2_score(y_te, yp), fi, y_te.values[:100], yp[:100], len(features), int(len(inmet)*0.8)

            mae, r2, fi, y_true, y_pred_m, n_feat, n_train = _train_for_pdf(inmet)

            m_data = [
                ["MAE", "R2", "Features", "Amostras treino"],
                [f"{mae:.2f} µg/m3", f"{r2:.3f}", str(n_feat), f"{n_train:,}"],
            ]
            t = Table(m_data, colWidths=[1.7*inch]*4)
            t.setStyle(tbl_style())
            story_elements.append(t)
            story_elements.append(Spacer(1, 8))

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

            # Feature importance
            ax = axes[0]
            fi_vals = fi.values
            fi_lbls = fi.index.tolist()
            colors_fi = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(fi_vals)))
            bars = ax.barh(fi_lbls[::-1], fi_vals[::-1], color=colors_fi, edgecolor="none")
            ax.set_title("Importancia das features")
            ax.set_xlabel("Importancia", color=TXT2)
            ax.grid(axis="x", alpha=0.3)

            # Real vs Previsto
            ax2 = axes[1]
            ax2.plot(y_true, color=C_TEAL, linewidth=1.5, label="Real", alpha=0.9)
            ax2.plot(y_pred_m, color=ACCENT, linewidth=1.5, linestyle="--", label="Previsto", alpha=0.9)
            ax2.set_title("Real vs Previsto — amostra 100 pontos")
            ax2.set_xlabel("Amostras", color=TXT2)
            ax2.set_ylabel("PM2.5 µg/m³", color=TXT2)
            ax2.legend(fontsize=8)
            ax2.grid(axis="y", alpha=0.3)

            for a in axes:
                a.set_facecolor(CARD)
            fig.tight_layout(pad=1.4)
            add_chart(fig, height=3.2*inch)
        else:
            story_elements.append(Paragraph("⚠ Dados INMET nao carregados — modelo indisponivel.", style_body))

        story_elements.append(PageBreak())

        # ══════════════════════════════════════════════════════════════════════════
        # SEÇÃO 7 — GEAR
        # ══════════════════════════════════════════════════════════════════════════
        story_elements.append(Paragraph("06 — RECOMENDACAO DE EQUIPAMENTO", style_tag))
        story_elements.append(Paragraph("Guia de Equipamento — MotoAR", style_h1))
        story_elements.append(Paragraph(
            "Logica de recomendacao baseada nos dados reais INMET 2025 + IQAir Mar 2026. "
            "Thresholds calibrados para Brasilia-DF.", style_body))

        gear_rows = [["Gear", "Nome", "Threshold", "Fonte", "Frequencia esperada", "Prio."]]
        for row in gear_data:
            icon, name, thresh, fonte, freq, prio, _ = row
            gear_rows.append([icon, name, thresh, fonte, freq, prio])
        cws_g = [0.35*inch, 1.7*inch, 1.8*inch, 1.1*inch, 1.35*inch, 0.55*inch]
        t = Table(gear_rows, colWidths=cws_g)
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(Spacer(1, 12))

        # Guia sazonal
        story_elements.append(Paragraph("Guia Sazonal de Brasilia", style_h2))
        seasonal_data = [
            ["Periodo", "Estacao", "PM2.5 tipico", "Recomendacao", "Nivel"],
            ["Jan-Jun",  "Estacao Chuvosa",       "4-9 µg/m3",  "Equipamento leve. Capa de chuva para pancadas vespertinas.", "Otimo"],
            ["Jul-Out",  "Seca / Queimadas",      "17-22 µg/m3","FILTRO OBRIGATORIO. 65% horas noturnas > OMS. Evitar 19-21h.", "Critico"],
            ["Nov-Dez",  "Transicao",             "9-12 µg/m3", "Chuvas voltam, mas irregularmente. Filtro recomendado.",      "Moderado"],
        ]
        cws_s = [0.7*inch, 1.3*inch, 0.9*inch, 3.3*inch, 0.65*inch]
        t = Table(seasonal_data, colWidths=cws_s)
        t.setStyle(tbl_style())
        story_elements.append(t)
        story_elements.append(Spacer(1, 10))

        # Gráfico % horas acima OMS por mês (se INMET disponível)
        if inmet is not None:
            monthly_over = inmet.groupby("month").apply(
                lambda x: (x["pm25"] > 15).sum() / len(x) * 100
            ).reset_index()
            monthly_over.columns = ["month","pct_acima"]

            fig, ax = plt.subplots(figsize=(10, 3))
            bar_c2 = [C_BLUE if m<=6 else C_RED if m<=10 else C_YEL for m in monthly_over["month"]]
            brs = ax.bar(MONTH_L[:len(monthly_over)], monthly_over["pct_acima"],
                         color=bar_c2, edgecolor="none")
            for b, v in zip(brs, monthly_over["pct_acima"]):
                ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%", ha="center",
                        va="bottom", fontsize=7, color=TXT2)
            ax.axhline(50, color=C_ORAN, linewidth=1.2, linestyle="--", label="50% das horas")
            ax.set_title("% de horas acima do limiar OMS (15 µg/m³) por mes")
            ax.set_ylabel("%", color=TXT2)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_facecolor(CARD)
            fig.tight_layout()
            add_chart(fig, height=2.6*inch)

        doc.build(story_elements)
        buffer.seek(0)
        return buffer.getvalue()

    st.markdown("""
    <table class="styled-table">
        <thead><tr><th>Gear</th><th>Nome</th><th>Threshold</th><th>Fonte</th><th>Frequência esperada*</th><th>Prio.</th></tr></thead>
        <tbody>
    """, unsafe_allow_html=True)
    for row in gear_data:
        icon,name,thresh,fonte,freq,prio,cls = row
        st.markdown(f"""
        <tr>
            <td style="font-size:20px;">{icon}</td>
            <td style="color:#F0F2F5;font-weight:500;">{name}</td>
            <td style="font-size:11px;">{thresh}</td>
            <td style="font-size:10px;color:#8A9099;">{fonte}</td>
            <td style="font-size:10px;">{freq}</td>
            <td><span class="badge {cls}">{prio}</span></td>
        </tr>""", unsafe_allow_html=True)
    st.markdown("</tbody></table>", unsafe_allow_html=True)

    st.markdown('<div class="sec-sub" style="margin-top:.5rem;">* Estimativas baseadas em dados INMET 2025 CRAS Fercal.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)

    st.markdown("#### 📅 Guia sazonal de Brasília")
    seasonal = [
        ("Janeiro – Junho", "🌧️ Estação Chuvosa", "PM2.5 ≈ 4–9 µg/m³", "Equipamento leve. Apenas capa de chuva para as frequentes pancadas vespertinas. Qualidade do ar boa.", COLORS["blue"], "Ótimo"),
        ("Julho – Outubro", "🔥 Estação Seca / Queimadas", "PM2.5 ≈ 17–22 µg/m³", "FILTRO OBRIGATÓRIO. PM2.5 ultrapassa meta OMS em 65% das horas noturnas. Evitar pico 19–21h. Hidratação extra.", COLORS["red"], "Crítico"),
        ("Novembro – Dezembro", "🍂 Transição", "PM2.5 ≈ 9–12 µg/m³", "Chuvas voltam, mas irregularmente. PM2.5 em queda mas ainda elevado comparado ao período chuvoso. Filtro recomendado.", COLORS["yellow"], "Moderado"),
    ]
    cols = st.columns(3)
    for col, (period, name, pm, desc, clr, label) in zip(cols, seasonal):
        with col:
            st.markdown(f"""
            <div style="background:#1E2126;border:2px solid {clr};border-radius:12px;padding:1.25rem;height:100%;">
                <div style="font-family:'Barlow Condensed';font-size:22px;font-weight:900;color:{clr};margin-bottom:4px;">{name}</div>
                <div style="font-family:'IBM Plex Mono';font-size:11px;color:#8A9099;margin-bottom:8px;">{period}</div>
                <div style="font-size:13px;font-weight:600;color:#F0F2F5;margin-bottom:8px;">{pm}</div>
                <div style="font-size:12px;color:#8A9099;line-height:1.6;">{desc}</div>
                <div style="margin-top:12px;"><span class="badge" style="background:rgba(255,255,255,.05);color:{clr};border:0.5px solid {clr};">{label}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
    st.markdown("#### 📄 Relatório Completo")
    st.markdown('<div class="sec-sub">Exporta todas as seções do MotoAR em PDF: visão geral, LCA, EDA INMET, EDA IQAir, cruzamentos, modelo preditivo e recomendações de gear.</div>', unsafe_allow_html=True)
    pdf_bytes = generate_full_report_pdf()
    st.download_button(
        label="⬇️ Baixar Relatório Completo (PDF)",
        data=pdf_bytes,
        file_name="motoar_relatorio_completo.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    if inmet is not None:
        st.markdown('<div class="hrdiv"></div>', unsafe_allow_html=True)
        st.markdown("#### % de horas acima do limiar OMS (15 µg/m³) por mês")
        monthly_over = inmet.groupby("month").apply(
            lambda x: (x["pm25"] > 15).sum() / len(x) * 100
        ).reset_index()
        monthly_over.columns = ["month","pct_acima"]
        monthly_over["month_name"] = monthly_over["month"].apply(lambda m: MONTH_LABELS[m-1])
        fig = px.bar(monthly_over, x="month_name", y="pct_acima",
                     color="pct_acima", color_continuous_scale=["#1A6B3C","#B8860B","#C8401A"],
                     labels={"month_name":"Mês","pct_acima":"% horas acima de 15 µg/m³"},
                     text=monthly_over["pct_acima"].round(1).astype(str)+"%")
        fig.update_traces(textposition="outside", textfont_size=9)
        fig.add_hline(y=50, line_dash="dash", line_color=COLORS["orange"],
                      annotation_text="50% das horas", annotation_font_size=9)
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig_layout(fig, 320), use_container_width=True)
