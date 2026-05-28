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
                # Converter hora para label de string para forçar distribuição categórica em 360°
                hs["hora_label"] = hs["hour"].apply(lambda h: f"{h:02d}h")
                # Ordem correta das 24 horas para o eixo categórico
                hora_order = [f"{h:02d}h" for h in range(24)]
                fig = px.line_polar(
                    hs,
                    r="pm25",
                    theta="hora_label",
                    color="season",
                    line_close=True,
                    color_discrete_map=season_colors,
                    template="plotly_dark",
                    start_angle=90,
                    direction="clockwise",
                    category_orders={"hora_label": hora_order},
                )
                fig.add_trace(go.Scatterpolar(
                    r=[15] * 24,
                    theta=hora_order,
                    mode='lines',
                    name='Limite OMS',
                    line=dict(color=colors["teal"], width=2, dash='dash'),
                    hoverinfo='skip' # Para não atrapalhar a leitura dos dados reais
                ))
                fig.update_traces(line_width=2.5)
                fig.update_layout(
                    polar=dict(
                        bgcolor=colors["card"],
                        radialaxis=dict(
                            visible=True,
                            range=[0, hs["pm25"].max() + 5],
                            gridcolor="rgba(255,255,255,0.12)",
                            tickfont=dict(color=colors["txt2"], size=9),
                        ),
                        angularaxis=dict(
                            direction="clockwise",
                            rotation=90,
                            gridcolor="rgba(255,255,255,0.1)",
                            tickfont=dict(color=colors["txt2"], size=9),
                        )
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=30, l=30, r=30),
                    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
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
            st.info("📊 Correlação PM2.5 × NO2: r = 0.47 — ambos relacionados à combustão (tráfego + queimadas)")
            

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
            st.info("Dias com chuva >5mm têm PM2.5 médio de 7,9 µg/m³ vs 12,8 µg/m³ nos dias secos — chuva melhora a qualidade do ar ao limpar partículas.")
            st.info("Limitação: chuva acumulada diária pode não capturar eventos de chuva rápida ou intermitente que também afetam PM2.5.")

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
                         color_continuous_scale="teal", 
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
            st.info("O modelo captura bem as tendências gerais, mas pode subestimar picos extremos de PM2.5 — importante considerar para alertas de saúde.")

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
            st.info("✅ Condições ideais para sair — qualidade do ar excelente")
        elif score >= 65:
            score_color, score_label = COLORS["blue"],   "Bom para sair"
            st.info("👍 Condições boas para sair — qualidade do ar dentro dos limites")    
        elif score >= 55:
            score_color, score_label = COLORS["yellow"], "Favorável"
            st.info("⚠️ Condições favoráveis, mas fique atento — qualidade do ar moderada")
        elif score >= 45:
            score_color, score_label = COLORS["orange"], "Use máscara N95"
            st.warning("⚠️ Qualidade do ar ruim — recomenda-se usar máscara N95")    
        elif score >= 35:
            score_color, score_label = COLORS["orange"], "Use filtro de ar"
            st.warning("⚠️ Qualidade do ar ruim — recomenda-se usar capacete com filtro de ar")
        else:
            score_color, score_label = COLORS["red"],    "Evite ou use EPI completo"
            st.error("❌ Condições desfavoráveis — risco para a saúde")


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
            if pm25_pred > 15: gear.append("⚠️ Atenção extra — PM2.5 acima do recomendado pela OMS")
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
        ["💧","Hidratação extra (2L+)","Temp > 30°C E Umidade < 40%","IQAir + INMET","Set–Nov: 30% dos dias secos","Média","b-yellow"],
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
        import matplotlib.dates as mdates
        from reportlab.platypus import PageBreak
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.pagesizes import A4
        # Cores e estilos

        BG      = "#0D0F12"
        CARD    = "#1A1D22"
        CARD2   = "#16191C"
        TXT     = "#F0F2F5"
        TXT2    = "#8A9099"
        ACCENT  = "#FF5722"
        C_RED   = "#E74C3C"
        C_RUBRO = "#C0392B"
        C_BLUE  = "#3498DB"
        C_GREEN = "#2ECC71"
        C_YEL   = "#F1C40F"
        C_TEAL  = "#1ABC9C"
        C_ORAN  = "#FF8C00"
        C_VELV  = "#7D3C98"
        C_RED2  = "#C0392B"
        C_BLUE2 = "#2980B9"
        C_GREEN2= "#27AE60"
        C_YEL2  = "#D4AC0D"
        C_TEAL2 = "#13A185"
        C_ORAN2 = "#E67E22"
        COLORS2 = [C_RED, C_BLUE, C_GREEN, C_YEL, C_TEAL, C_ORAN, C_VELV]
        
        
        # Mapas de estações e sensores
        INMET_STATIONS = {
            "Brasília-DF": {"lat": -15.793889, "lon": -47.882778},
            "Taguatinga-DF": {"lat": -15.835, "lon": -48.08},
            "Gama-DF": {"lat": -15.936, "lon": -48.05},
            "Planaltina-DF": {"lat": -15.58, "lon": -47.65},
        }
        IQAIR_SENSORS = {
            "Sensor A": {"lat": -15.8, "lon": -47.9},
            "Sensor B": {"lat": -15.9, "lon": -48.0},
            "Sensor C": {"lat": -15.7, "lon": -48.1},
            "Sensor D": {"lat": -15.85, "lon": -47.95},
        }
        # Cores para estações e sensores
        STATION_COLORS = {
            "Brasília-DF": COLORS["C_red2"],
            "Taguatinga-DF": COLORS["C_blue2"],
            "Gama-DF": COLORS["C_green2"],
            "Planaltina-DF": COLORS["C_yellow2"],
        }
        SENSOR_COLORS = {
            "Sensor A": COLORS["C_RUBRO"],
            "Sensor B": COLORS["C_velv"],
            "Sensor C": COLORS["C_teal"],
            "Sensor D": COLORS["C_oran"],
        }           
        # Rótulos de meses
        
        MONTH_L = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        MONTH_LABELS = {i+1: m for i,m in enumerate(MONTH_L)}
        
        # Dados estáticos para o relatório (médias mensais e horários típicos)
        ## EXPLICAÇÃO: Esses dados foram extraídos dos datasets INMET 2025 e IQAir Mar 2026, calculando as médias mensais de PM2.5 e os padrões horários típicos de AQI, chuva e temperatura. Eles servem como referência para o relatório, mas têm limitações por serem baseados em um único ano e cidade (Brasília-DF), podendo não refletir outras regiões ou anos com padrões climáticos diferentes.

        STATIC_MONTHLY_PM   = [5.1,7.13,9.49,6.07,4.2,5.84,17.77,18.3,22.25,18.82,12.12,9.44]
        STATIC_HOURLY_SECA  = [28.09,24.93,21.19,19.01,17.46,16.61,16.33,17.27,21.22,23.51,18.0,9.91,8.24,7.52,7.97,7.06,8.91,12.58,16.86,26.69,34.0,33.43,31.91,30.72]
        STATIC_HOURLY_CHUVA = [7.23,5.77,4.81,4.12,4.11,3.82,4.11,6.32,8.35,8.21,5.64,3.83,3.66,3.9,4.21,4.99,5.05,5.91,6.55,9.28,11.53,10.62,8.59,8.2]
        STATIC_HOURLY_TRANS = [8.83,7.66,6.92,7.8,8.45,9.12,11.25,15.8,13.22,9.04,6.72,8.09,6.07,4.54,6.4,6.27,9.33,10.74,15.22,21.58,20.31,16.98,14.74,11.97]
        STATIC_PCT_OMS      = [5.2,8.3,14.1,7.6,3.9,6.2,61.4,65.8,74.3,63.1,38.7,22.5]
        STATIC_NOTE         = "Dados INMET 2025 + IQAir Mar 2026 — valores médios mensais e horários típicos para Brasília-DF. Limitação: dados de apenas um ano e uma cidade, podem não refletir outras regiões ou anos."

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": CARD,
            "text.color": TXT, "axes.labelcolor": TXT2,
            "xtick.color": TXT2, "ytick.color": TXT2,
            "axes.edgecolor": "#2C3038", "grid.color": "#252930",
            "grid.linewidth": 0.5, "axes.titlesize": 11,
            "axes.titlecolor": TXT, "axes.titleweight": "bold",
            "font.family": "monospace", "legend.framealpha": 0,
            "legend.labelcolor": TXT2,
        })

        def fig_to_image(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                        facecolor=BG, edgecolor="none")
            buf.seek(0)
            plt.close(fig)
            return buf

        story = []
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=40, rightMargin=40,
                                topMargin=40, bottomMargin=40)
        W = A4[0] - 80

        rl_styles = getSampleStyleSheet()
        s_tag  = ParagraphStyle("tag",  fontSize=8,  textColor=colors.HexColor(ACCENT),
                                spaceAfter=2, fontName="Helvetica-Bold", leading=10)
        s_h1   = ParagraphStyle("h1",   fontSize=20, textColor=colors.HexColor(ACCENT),
                                spaceBefore=14, spaceAfter=5, fontName="Helvetica-Bold")
        s_h2   = ParagraphStyle("h2",   fontSize=12, textColor=colors.HexColor(TXT),
                                spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
        s_body = ParagraphStyle("body", fontSize=8.5,textColor=colors.HexColor("#C0C4CC"),
                                leading=13, spaceAfter=5)
        s_cov  = ParagraphStyle("cov",  fontSize=34, textColor=colors.HexColor(ACCENT),
                                fontName="Helvetica-Bold", spaceAfter=4)
        s_cov2 = ParagraphStyle("cov2", fontSize=11, textColor=colors.HexColor(TXT2),
                                spaceAfter=6)
        s_note = ParagraphStyle("note", fontSize=7.5,textColor=colors.HexColor(TXT2),
                                leading=11, spaceAfter=3)

        def tbl_style():
            return TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor(ACCENT)),
                ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
                ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1),  8),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                    [colors.HexColor(CARD), colors.HexColor(CARD2)]),
                ("TEXTCOLOR",     (0,1), (-1,-1),  colors.HexColor("#C0C4CC")),
                ("GRID",          (0,0), (-1,-1),  0.3, colors.HexColor("#2C3038")),
                ("LEFTPADDING",   (0,0), (-1,-1),  6),
                ("RIGHTPADDING",  (0,0), (-1,-1),  6),
                ("TOPPADDING",    (0,0), (-1,-1),  4),
                ("BOTTOMPADDING", (0,0), (-1,-1),  4),
                ("VALIGN",        (0,0), (-1,-1),  "MIDDLE"),
            ])

        def add_img(fig, w=W, h=None):
            img_buf = fig_to_image(fig)
            story.append(Image(img_buf, width=w, height=h or w*0.38))
            story.append(Spacer(1, 6))

        def sec(tag, title, sub=""):
            story.append(Paragraph(tag, s_tag))
            story.append(Paragraph(title, s_h1))
            if sub:
                story.append(Paragraph(sub, s_body))

        # ── CAPA ─────────────────────────────────────────────────────────────────
        story.append(Spacer(1, 60))
        story.append(Paragraph("MotoAR", s_cov))
        story.append(Paragraph("Análise de Qualidade do Ar para Motociclistas", s_cov2))
        story.append(Paragraph("Brasília-DF · INMET 2025 + IQAir Mar 2026 · XGBoost", s_cov2))
        story.append(Spacer(1, 20))

        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        bc = [C_BLUE if i<6 else C_RED if i<10 else C_YEL for i in range(12)]
        bars = ax.bar(MONTH_L, STATIC_MONTHLY_PM, color=bc, edgecolor="none", width=0.7)
        ax.axhline(15, color=C_ORAN, lw=1.5, ls="--", alpha=0.9, label="OMS 24h — 15 µg/m³")
        for b, v in zip(bars, STATIC_MONTHLY_PM):
            ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=8, color=TXT2)
        ax.set_title("PM2.5 médio mensal — CRAS Fercal 2025", color=TXT, fontsize=12, pad=8)
        ax.set_ylabel("µg/m³", color=TXT2); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        add_img(fig, h=2.5*inch)

        story.append(Spacer(1, 8))
        mt = Table([["Registros Totais","Parâmetros","Sensores","Cobertura","Acurácia Modelo"],
                    ["45.061","18","6","14 meses","~82%"]], colWidths=[W/5]*5)
        mt.setStyle(tbl_style()); story.append(mt)
        story.append(PageBreak())

        # ── 00 VISÃO GERAL ────────────────────────────────────────────────────────
        sec("00 — DASHBOARD", "Visão Geral do Projeto",
            "App de qualidade do ar para motociclistas de Brasília. Combina dados INMET "
            "(estação fixa horária) com IQAir (4 sensores quasi-tempo real) para recomendar "
            "equipamentos e horários de saída.")

        horas = list(range(24))
        fig = plt.figure(figsize=(12, 4))
        fig.patch.set_facecolor(BG)
        ax1 = fig.add_subplot(121); ax1.set_facecolor(CARD)
        ax1.plot(horas, STATIC_HOURLY_SECA,  color=C_RED,  lw=2.5, label="Seca (Jul-Out)")
        ax1.fill_between(horas, STATIC_HOURLY_SECA,  alpha=0.1,  color=C_RED)
        ax1.plot(horas, STATIC_HOURLY_TRANS, color=C_YEL,  lw=1.8, ls="--", label="Transição")
        ax1.plot(horas, STATIC_HOURLY_CHUVA, color=C_BLUE, lw=2.5, label="Chuva (Jan-Jun)")
        ax1.fill_between(horas, STATIC_HOURLY_CHUVA, alpha=0.08, color=C_BLUE)
        ax1.axhline(15, color=C_ORAN, lw=1.2, ls=":", alpha=0.8, label="OMS 15")
        ax1.set_title("Padrão horário PM2.5 por estação")
        ax1.set_xlabel("Hora do dia", color=TXT2); ax1.set_ylabel("PM2.5 µg/m³", color=TXT2)
        ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.25)

        ax2 = fig.add_subplot(122, projection="polar"); ax2.set_facecolor(CARD)
        theta = [h*(2*np.pi/24) for h in horas] + [0]
        for vals, clr, lbl in [(STATIC_HOURLY_SECA,C_RED,"Seca"),
                                (STATIC_HOURLY_TRANS,C_YEL,"Transição"),
                                (STATIC_HOURLY_CHUVA,C_BLUE,"Chuva")]:
            r = list(vals)+[vals[0]]
            ax2.plot(theta, r, color=clr, lw=2.2, label=lbl)
            ax2.fill(theta, r, alpha=0.07, color=clr)
        ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
        ax2.set_xticks([h*(2*np.pi/24) for h in range(0,24,3)])
        ax2.set_xticklabels([f"{h}h" for h in range(0,24,3)], fontsize=8, color=TXT2)
        ax2.tick_params(colors=TXT2); ax2.yaxis.label.set_color(TXT2)
        ax2.set_title("PM2.5 — relógio 24h", color=TXT, fontsize=10, pad=12)
        ax2.legend(fontsize=7, loc="lower right")
        fig.tight_layout(pad=1.5)
        add_img(fig, h=3.1*inch)

        story.append(Paragraph("Achados Principais", s_h2))
        t = Table([
            ["Achado","Detalhe"],
            ["Amplitude sazonal 5.3×","PM2.5 vai de 4,2 µg/m³ (mai) a 22,3 µg/m³ (set). Queimadas dominam."],
            ["Dois picos diários","7–9h (tráfego) e 19–21h (inversão térmica). Confirmado INMET e IQAir."],
            ["Chuva reduz PM2.5 34%","Dias >5mm chuva: PM2.5 médio 7,9 vs 12,8 nos dias secos."],
            ["70% diferença espacial","UnB Gama tem AQI 70% maior que Finatec. Rota importa tanto quanto hora."],
            ["NO2 correlaciona PM2.5","r=0.47. NO2 como proxy de combustão (tráfego + queimadas)."],
            ["65% horas exigem filtro","Estação seca: 65% das horas noturnas > limiar OMS de 15 µg/m³."],
        ], colWidths=[2.2*inch, W-2.2*inch])
        t.setStyle(tbl_style()); story.append(t)
        story.append(PageBreak())

        # ── 01 LCA ────────────────────────────────────────────────────────────────
        sec("01 — CICLO DE VIDA", "LCA — MotoAR",
            "Mapeamento completo do ciclo de vida dos dados: da coleta bruta ao MotoAR Score "
            "entregue ao motociclista.")

        fig, ax = plt.subplots(figsize=(12, 2.2))
        fig.patch.set_facecolor(BG); ax.axis("off"); ax.set_facecolor(BG)
        stages = ["01\nIngestão","02\nLimpeza","03\nFeature\nEng.","04\nModelo","05\nEntrega"]
        xs = np.linspace(0.08, 0.92, 5)
        for i, (x, lbl) in enumerate(zip(xs, stages)):
            p = mpatches.FancyBboxPatch((x-0.07,0.1),0.14,0.8,
                boxstyle="round,pad=0.02",facecolor=CARD,edgecolor=ACCENT,linewidth=1.5)
            ax.add_patch(p)
            ax.text(x,0.5,lbl,ha="center",va="center",fontsize=9,color=TXT,
                    fontweight="bold",multialignment="center")
            if i<4:
                ax.annotate("",xy=(xs[i+1]-0.075,0.5),xytext=(x+0.075,0.5),
                            arrowprops=dict(arrowstyle="->",color=ACCENT,lw=1.5))
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        fig.tight_layout()
        add_img(fig, h=1.5*inch)

        t = Table([
            ["Etapa","Nome","Ações principais"],
            ["01","Ingestão","IQAir API · INMET estações automáticas · Open-Meteo forecast · Polling 5min"],
            ["02","Limpeza","Filtro status Ok · Clipping outliers · Imputação nulos · Normalização unidades"],
            ["03","Feature Eng.","Hora sin/cos · Flag seca/chuva · Rolling avg 3h/24h · Chuva acumulada 6h"],
            ["04","Modelo","XGBoost · Índice Saída 0-100 · Previsão PM2.5 +6h · Classificação risco"],
            ["05","Entrega","Push notification · Tela Agora MotoAR · Alerta queimadas Jul–Out · Gear"],
        ], colWidths=[0.5*inch,1.1*inch,W-1.6*inch])
        t.setStyle(tbl_style()); story.append(t)
        story.append(Spacer(1,6))

        story.append(Paragraph("Inventário de Dados", s_h2))
        t = Table([
            ["Fonte","Cobertura","Registros","Parâmetros","Freq.","Status"],
            ["INMET — CRAS Fercal","Jan-Dez 2025","8.760","14 params","1h","Ativo ✓"],
            ["INMET — Estação Escola","Jan-Dez 2025","8.760","8 params","1h","Ativo ✓"],
            ["IQAir — Brasilia centro","Mar 2026","9.853","AQI+PM2.5","~1min","Live"],
            ["IQAir — Escola 115 Norte","Mar 2026","8.766","AQI+PM2.5","~1min","Live"],
            ["IQAir — UnB Gama","Mar 2026","9.053","AQI+PM2.5","~1min","Live"],
            ["IQAir — Finatec","Mar 2026","8.629","AQI+PM2.5","~1min","Live"],
            ["Open-Meteo (planejado)","Futuro","—","Temp+Precip","1h","Integrar"],
            ["INMET 2024 (planejado)","Jan-Dez 2024","8.784","Mesmos","1h","Pendente"],
        ], colWidths=[2.0*inch,1.0*inch,0.8*inch,0.9*inch,0.55*inch,0.7*inch])
        t.setStyle(tbl_style()); story.append(t)
        story.append(Spacer(1,6))

        story.append(Paragraph("Limitações Identificadas", s_h2))
        t = Table([
            ["Limitação","Impacto"],
            ["INMET — sensor temperatura inválido","Valores negativos extremos filtrados"],
            ["IQAir — apenas mar/2026","Série curta; ideal ≥ 6 meses"],
            ["INMET 2024 — não recebido ainda","Reduz capacidade de treino do modelo"],
            ["Cobertura espacial limitada","6 sensores para toda Brasília-DF"],
            ["PM2.5 IQAir em escala AQI","Não diretamente comparável com INMET µg/m³"],
        ], colWidths=[3.0*inch,W-3.0*inch])
        t.setStyle(tbl_style()); story.append(t)
        story.append(PageBreak())

        # ── 02 EDA INMET ─────────────────────────────────────────────────────────
        sec("02 — ANÁLISE EXPLORATÓRIA", "EDA — INMET 2025",
            "Estação CRAS Fercal · Série horária completa Jan–Dez 2025 · 14 parâmetros ambientais.")

        inmet_data = inmet
        using_static = inmet_data is None
        if using_static:
            rows = []
            for m in range(1,13):
                season = "🌧️ Chuva" if m<=6 else "🔥 Seca" if m<=10 else "🍂 Transição"
                hs_v = STATIC_HOURLY_SECA if season=="🔥 Seca" else (STATIC_HOURLY_CHUVA if season=="🌧️ Chuva" else STATIC_HOURLY_TRANS)
                for h in range(24):
                    pm = hs_v[h]
                    rows.append({"month":m,"hour":h,"pm25":pm,"season":season,
                                 "no2":pm*1.3+2,"co":pm*0.04,"o3":80-pm*0.8,
                                 "so2":pm*0.3,"rain":0.5 if m<=6 else 0})
            inmet_data = pd.DataFrame(rows)

        met = Table([
            ["Registros PM2.5","PM2.5 médio","PM2.5 máx.","Mediana","Desvio padrão"],
            [f"{len(inmet_data):,}",f"{inmet_data['pm25'].mean():.1f} µg/m³",
             f"{inmet_data['pm25'].max():.1f} µg/m³",f"{inmet_data['pm25'].median():.1f} µg/m³",
             f"{inmet_data['pm25'].std():.1f} µg/m³"],
        ], colWidths=[W/5]*5)
        met.setStyle(tbl_style()); story.append(met)
        story.append(Spacer(1,8))

        fig, axes = plt.subplots(2,2,figsize=(12,7.5)); fig.patch.set_facecolor(BG)
        s_map = {"🌧️ Chuva":C_BLUE,"🔥 Seca":C_RED,"🍂 Transição":C_YEL}

        ax = axes[0][0]; ax.set_facecolor(CARD)
        monthly = inmet_data.groupby("month")["pm25"].agg(["mean",lambda x:x.quantile(.75),"max"]).reset_index()
        monthly.columns = ["month","mean","p75","max"]
        bc2 = [C_RED if 7<=m<=10 else C_BLUE if m<=6 else C_YEL for m in monthly["month"]]
        ax.bar(MONTH_L[:len(monthly)],monthly["mean"],color=bc2,edgecolor="none",label="Média")
        ax.plot(MONTH_L[:len(monthly)],monthly["p75"],color=C_ORAN,lw=1.8,ls="--",marker="o",ms=4,label="P75")
        ax.plot(MONTH_L[:len(monthly)],monthly["max"],color=C_RED,ls="",marker="x",ms=5,label="Máx")
        ax.axhline(15,color=C_ORAN,lw=1,ls=":",alpha=0.6)
        ax.legend(fontsize=7); ax.grid(axis="y",alpha=0.25)
        ax.set_title("PM2.5 mensal — média, P75, máximo"); ax.set_ylabel("µg/m³",color=TXT2)

        ax = axes[0][1]; ax.set_facecolor(CARD)
        for season,sc in s_map.items():
            sub = inmet_data[(inmet_data["season"]==season)&(inmet_data["pm25"]<100)]["pm25"]
            if len(sub):
                ax.hist(sub,bins=30,alpha=0.65,color=sc,label=" ".join(season.split()[:2]),edgecolor="none")
        ax.axvline(15,color=C_ORAN,lw=1.5,ls="--",label="OMS 15")
        ax.set_title("Distribuição PM2.5 por estação"); ax.set_xlabel("PM2.5 µg/m³",color=TXT2)
        ax.legend(fontsize=7); ax.grid(axis="y",alpha=0.25)

        ax = axes[1][0]; ax.set_facecolor(CARD)
        hs = inmet_data.groupby(["hour","season"])["pm25"].mean().reset_index()
        for season,sc in s_map.items():
            sub = hs[hs["season"]==season]
            if len(sub):
                ax.plot(sub["hour"],sub["pm25"],color=sc,lw=2.5,label=" ".join(season.split()[:2]))
        ax.axhline(15,color=C_GREEN,lw=1,ls="--",alpha=0.7,label="OMS 24h")
        ax.set_title("Padrão horário por estação")
        ax.set_xlabel("Hora",color=TXT2); ax.set_ylabel("PM2.5 µg/m³",color=TXT2)
        ax.legend(fontsize=7); ax.grid(axis="y",alpha=0.25)

        ax = axes[1][1]; ax.set_facecolor(CARD)
        pivot = inmet_data.groupby(["hour","month"])["pm25"].mean().unstack(fill_value=0)
        im = ax.imshow(pivot.values,aspect="auto",cmap="YlOrRd",
                       extent=[-0.5,pivot.shape[1]-0.5,23.5,-0.5],origin="upper")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(MONTH_L[:pivot.shape[1]],fontsize=7)
        ax.set_ylabel("Hora",color=TXT2); ax.set_title("Heatmap PM2.5 — hora × mês")
        plt.colorbar(im,ax=ax,shrink=0.8,label="µg/m³")

        fig.tight_layout(pad=1.5)
        add_img(fig, h=4.8*inch)

        if using_static:
            story.append(Paragraph("* Dados estáticos representativos (arquivo INMET não carregado).", s_note))

        story.append(Paragraph("Estatísticas Descritivas — Poluentes", s_h2))
        cols_s = ["pm25","no2","co","o3","so2","rain"]
        lbl_s  = ["PM2.5 µg/m³","NO2 µg/m³","CO ppm","O3 µg/m³","SO2 µg/m³","Chuva mm"]
        stats  = inmet_data[cols_s].describe().T.round(2)
        sr = [["Parâmetro","Count","Média","Std","Mín","P25","P50","P75","Máx"]]
        for lbl,row in zip(lbl_s,stats.itertuples()):
            sr.append([lbl,str(int(row.count)),f"{row.mean:.2f}",f"{row.std:.2f}",
                       f"{row.min:.2f}",f"{getattr(row,'25%'):.2f}",
                       f"{getattr(row,'50%'):.2f}",f"{getattr(row,'75%'):.2f}",f"{row.max:.2f}"])
        t = Table(sr, colWidths=[1.4*inch]+[0.68*inch]*8)
        t.setStyle(tbl_style()); story.append(t)
        story.append(PageBreak())

        # ── 03 EDA IQAIR ─────────────────────────────────────────────────────────
        sec("03 — IQAIR · 4 SENSORES · MAR 2026", "EDA — IQAir Brasília",
            "36.301 registros de quasi-tempo real. 4 sensores distribuídos em Brasília.")

        if iq is not None:
            t = Table([
                ["Registros","AQI médio","AQI máximo","PM2.5 médio"],
                [f"{len(iq):,}",f"{iq['aqi'].mean():.1f}",f"{iq['aqi'].max()}",f"{iq['pm25'].mean():.2f}"],
            ], colWidths=[W/4]*4)
            t.setStyle(tbl_style()); story.append(t); story.append(Spacer(1,8))

            fig,axes = plt.subplots(2,2,figsize=(12,7.5)); fig.patch.set_facecolor(BG)
            sc4 = [C_GREEN,C_BLUE,C_YEL,C_RED]
            sensors = iq["sensor_location"].unique().tolist()

            ax=axes[0][0]; ax.set_facecolor(CARD)
            bp=ax.boxplot([iq[iq["sensor_location"]==s]["aqi"].dropna().values for s in sensors],
                          patch_artist=True,widths=0.55,
                          medianprops=dict(color=TXT,linewidth=1.8),
                          flierprops=dict(marker=".",color=TXT2,markersize=3))
            for patch,sc in zip(bp["boxes"],sc4):
                patch.set_facecolor(sc); patch.set_alpha(0.75)
            ax.set_xticklabels([s[:14] for s in sensors],fontsize=7,rotation=8)
            ax.set_title("AQI por sensor — box plot"); ax.set_ylabel("AQI",color=TXT2)
            ax.grid(axis="y",alpha=0.25)

            ax=axes[0][1]; ax.set_facecolor(CARD)
            daily=iq.groupby(["date","sensor_location"])["aqi"].mean().reset_index()
            daily["date"]=pd.to_datetime(daily["date"])
            for s,sc in zip(sensors,sc4):
                sub=daily[daily["sensor_location"]==s]
                ax.plot(sub["date"],sub["aqi"],color=sc,lw=1.8,label=s[:14])
            ax.set_title("AQI diário — todos os sensores"); ax.legend(fontsize=6.5)
            ax.grid(axis="y",alpha=0.25)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            plt.setp(ax.xaxis.get_majorticklabels(),rotation=25,fontsize=7)

            ax=axes[1][0]; ax.set_facecolor(CARD)
            hourly_s=iq.groupby(["hour","sensor_location"])["aqi"].mean().reset_index()
            for s,sc in zip(sensors,sc4):
                sub=hourly_s[hourly_s["sensor_location"]==s]
                ax.plot(sub["hour"],sub["aqi"],color=sc,lw=2.2,label=s[:14])
            ax.set_title("Padrão horário AQI por sensor")
            ax.set_xlabel("Hora",color=TXT2); ax.set_ylabel("AQI médio",color=TXT2)
            ax.legend(fontsize=6.5); ax.grid(axis="y",alpha=0.25)

            ax=axes[1][1]; ax.set_facecolor(CARD)
            d2=iq.groupby("date").agg(aqi=("aqi","mean"),rh=("humidity","mean"),temp=("temperature","mean")).reset_index()
            sc_p=ax.scatter(d2["rh"],d2["aqi"],c=d2["temp"],cmap="RdYlGn_r",alpha=0.8,s=50,edgecolors="none")
            plt.colorbar(sc_p,ax=ax,shrink=0.8,label="Temp °C")
            ax.set_title("AQI × Umidade relativa")
            ax.set_xlabel("Umidade %",color=TXT2); ax.set_ylabel("AQI médio",color=TXT2)
            ax.grid(alpha=0.25)

            fig.tight_layout(pad=1.5); add_img(fig,h=4.8*inch)

            story.append(Paragraph("Estatísticas por Sensor", s_h2))
            ss=iq.groupby("sensor_location").agg(
                R=("aqi","count"),Am=("aqi","mean"),Ax=("aqi","max"),
                As=("aqi","std"),P=("pm25","mean"),T=("temperature","mean"),U=("humidity","mean")
            ).round(2).reset_index()
            sr2=[["Sensor","Registros","AQI Méd","AQI Máx","AQI Std","PM2.5","Temp","Umid"]]
            for _,row in ss.iterrows():
                sr2.append([str(row["sensor_location"])[:22],str(int(row["R"])),
                            f"{row['Am']:.1f}",f"{row['Ax']:.0f}",f"{row['As']:.1f}",
                            f"{row['P']:.2f}",f"{row['T']:.1f}",f"{row['U']:.1f}"])
            t=Table(sr2,colWidths=[2.1*inch]+[0.73*inch]*7)
            t.setStyle(tbl_style()); story.append(t)
        else:
            story.append(Paragraph("⚠ Arquivo IQAir CSV não carregado — carregue na barra lateral.", s_body))
        story.append(PageBreak())

        # ── 04 CRUZAMENTOS ────────────────────────────────────────────────────────
        sec("04 — CRUZAMENTO DOS DATASETS", "Cruzamentos INMET × IQAir",
            "Análise de correlações entre variáveis INMET e IQAir. "
            "Cada cruzamento valida uma hipótese do modelo preditivo.")

        fig,axes=plt.subplots(2,2,figsize=(12,7.5)); fig.patch.set_facecolor(BG)

        ax=axes[0][0]; ax.set_facecolor(CARD)
        cc=["pm25","no2","co","o3","rain"]; cl=["PM2.5","NO2","CO","O3","Chuva"]
        cm_v=inmet_data[cc].corr().values
        im=ax.imshow(cm_v,cmap="RdBu_r",vmin=-1,vmax=1)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(cl,fontsize=9); ax.set_yticklabels(cl,fontsize=9)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{cm_v[i,j]:.2f}",ha="center",va="center",fontsize=8.5,color=TXT)
        plt.colorbar(im,ax=ax,shrink=0.85); ax.set_title("Correlação — INMET")

        ax=axes[0][1]; ax.set_facecolor(CARD)
        for season,sc in {"🌧️ Chuva":C_BLUE,"🔥 Seca":C_RED,"🍂 Transição":C_YEL}.items():
            sub=inmet_data[(inmet_data["season"]==season)&(inmet_data["pm25"]<100)]
            sub=sub.sample(min(250,len(sub)),random_state=42)
            ax.scatter(sub["no2"],sub["pm25"],alpha=0.5,color=sc,s=12,
                      label=" ".join(season.split()[:2]),edgecolors="none")
        ax.set_title("Scatter PM2.5 × NO2")
        ax.set_xlabel("NO2 µg/m³",color=TXT2); ax.set_ylabel("PM2.5 µg/m³",color=TXT2)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

        ax=axes[1][0]; ax.set_facecolor(CARD)
        r_cats=["0mm (seco)","1-5mm (leve)","5-20mm (mod.)","≥20mm (forte)"]
        pm_v=[12.8,10.1,8.4,7.9]
        if inmet is not None:
            di=inmet.groupby("date").agg(pm25=("pm25","mean"),rain=("rain","sum")).reset_index()
            di["rc"]=pd.cut(di["rain"],[-1,0,5,20,1e9],labels=r_cats)
            re=di.groupby("rc",observed=True)["pm25"].mean()
            pm_v=re.values.tolist(); r_cats=re.index.astype(str).tolist()
        bs=ax.bar(r_cats,pm_v,color=[C_RED,C_YEL,C_TEAL,C_BLUE][:len(pm_v)],edgecolor="none")
        for b,v in zip(bs,pm_v):
            ax.text(b.get_x()+b.get_width()/2,v+0.2,f"{v:.1f}",ha="center",fontsize=9,color=TXT2)
        ax.set_title("Efeito da chuva sobre PM2.5")
        ax.set_xlabel("Chuva acumulada diária",color=TXT2); ax.set_ylabel("PM2.5 médio µg/m³",color=TXT2)
        ax.grid(axis="y",alpha=0.25)

        ax=axes[1][1]; ax.set_facecolor(CARD)
        inm_h=inmet_data.groupby("hour")["pm25"].mean()
        ax.plot(inm_h.index,inm_h.values,color=C_RED,lw=2.5,label="INMET PM2.5")
        ax.fill_between(inm_h.index,inm_h.values,alpha=0.08,color=C_RED)
        ax.set_ylabel("PM2.5 µg/m³",color=C_RED); ax.set_xlabel("Hora do dia",color=TXT2)
        if iq is not None:
            axb=ax.twinx(); axb.set_facecolor(CARD)
            iqh=iq.groupby("hour")["aqi"].mean()
            axb.plot(iqh.index,iqh.values,color=C_BLUE,lw=2,ls="--",label="IQAir AQI")
            axb.set_ylabel("AQI",color=C_BLUE)
            l1,la1=ax.get_legend_handles_labels(); l2,la2=axb.get_legend_handles_labels()
            ax.legend(l1+l2,la1+la2,fontsize=7.5)
        ax.set_title("Padrão horário: INMET vs IQAir"); ax.grid(axis="y",alpha=0.25)

        fig.tight_layout(pad=1.5); add_img(fig,h=4.8*inch)

        if iq is not None:
            diq=iq.groupby("date").agg(aqi=("aqi","mean"),temp=("temperature","mean")).reset_index()
            rc=diq[["temp","aqi"]].corr().iloc[0,1]
            story.append(Paragraph(f"📊 Correlação Temperatura × AQI: r = {rc:.3f} — temperatura mais baixa associada a AQI menor (ar úmido = melhor dispersão).", s_body))
        story.append(PageBreak())

        # ── 05 MODELO ─────────────────────────────────────────────────────────────
        sec("05 — MACHINE LEARNING", "Modelo Preditivo XGBoost",
            "XGBoost treinado com dados INMET 2025. Previsão de PM2.5 e cálculo do "
            "Índice de Saída 0–100 para o motociclista.")

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        @st.cache_resource
        def _pdf_model(df_raw):
            df=df_raw.copy()
            df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24)
            df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
            df["month_sin"]=np.sin(2*np.pi*df["month"]/12)
            df["month_cos"]=np.cos(2*np.pi*df["month"]/12)
            df["is_dry"]=(df["month"].between(7,10)).astype(int)
            df["pm25_lag1"]=df["pm25"].shift(1).fillna(df["pm25"].mean())
            df["pm25_lag3"]=df["pm25"].shift(3).fillna(df["pm25"].mean())
            df["pm25_roll3"]=df["pm25"].rolling(3,min_periods=1).mean()
            df["rain_acc6"]=df["rain"].rolling(6,min_periods=1).sum()
            feats=["hour_sin","hour_cos","month_sin","month_cos","is_dry",
                   "pm25_lag1","pm25_lag3","pm25_roll3","rain_acc6","no2","co"]
            dm=df[feats+["pm25"]].dropna()
            X,y=dm[feats],dm["pm25"]
            Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,random_state=42)
            try:
                from xgboost import XGBRegressor
                mdl=XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.05,random_state=42,verbosity=0)
            except ImportError:
                mdl=GradientBoostingRegressor(n_estimators=200,max_depth=5,random_state=42)
            mdl.fit(Xt,yt); yp=mdl.predict(Xe)
            fi=pd.Series(mdl.feature_importances_,index=feats).sort_values(ascending=False)
            return mean_absolute_error(ye,yp),r2_score(ye,yp),fi,ye.values[:100],yp[:100],len(feats),int(len(df)*0.8)

        mae_v,r2_v,fi_v,yt_v,yp_v,nf_v,nt_v=_pdf_model(inmet_data)
        t=Table([["MAE","R²","Features","Amostras treino"],
                 [f"{mae_v:.2f} µg/m³",f"{r2_v:.3f}",str(nf_v),f"{nt_v:,}"]],
                colWidths=[W/4]*4)
        t.setStyle(tbl_style()); story.append(t); story.append(Spacer(1,8))

        fig,axes=plt.subplots(1,2,figsize=(12,4)); fig.patch.set_facecolor(BG)
        ax=axes[0]; ax.set_facecolor(CARD)
        cf=plt.cm.YlOrRd(np.linspace(0.3,0.95,len(fi_v)))
        ax.barh(fi_v.index.tolist()[::-1],fi_v.values[::-1],color=cf,edgecolor="none")
        ax.set_title("Importância das features"); ax.set_xlabel("Importância",color=TXT2)
        ax.grid(axis="x",alpha=0.25)

        ax=axes[1]; ax.set_facecolor(CARD)
        ax.plot(yt_v,color=C_TEAL,lw=1.8,label="Real",alpha=0.9)
        ax.plot(yp_v,color=ACCENT,lw=1.8,ls="--",label="Previsto",alpha=0.9)
        ax.fill_between(range(len(yt_v)),yt_v,yp_v,alpha=0.07,color=C_ORAN)
        ax.set_title("Real vs Previsto — 100 pontos")
        ax.set_xlabel("Amostras",color=TXT2); ax.set_ylabel("PM2.5 µg/m³",color=TXT2)
        ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.25)

        fig.tight_layout(pad=1.5); add_img(fig,h=3.2*inch)
        if using_static:
            story.append(Paragraph("* Modelo treinado com dados estáticos representativos.", s_note))
        story.append(PageBreak())

        # ── 06 GEAR ──────────────────────────────────────────────────────────────
        sec("06 — RECOMENDAÇÃO DE EQUIPAMENTO", "Guia de Equipamento — MotoAR",
            "Lógica de recomendação baseada nos dados reais INMET 2025 + IQAir Mar 2026. "
            "Thresholds calibrados para Brasília-DF.")

        gr=[["","Nome","Threshold","Fonte","Frequência esperada","Prio."]]
        for row in gear_data:
            icon,name,thresh,fonte,freq,prio,_=row
            gr.append([icon,name,thresh,fonte,freq,prio])
        t=Table(gr,colWidths=[0.35*inch,1.6*inch,1.7*inch,1.1*inch,1.3*inch,0.65*inch])
        t.setStyle(tbl_style()); story.append(t); story.append(Spacer(1,8))

        story.append(Paragraph("Guia Sazonal de Brasília", s_h2))
        t=Table([
            ["Período","Estação","PM2.5","Recomendação","Nível"],
            ["Jan–Jun","Estação Chuvosa","4–9 µg/m³","Equipamento leve. Capa de chuva para pancadas vespertinas.","Ótimo"],
            ["Jul–Out","Seca / Queimadas","17–22 µg/m³","FILTRO OBRIGATÓRIO. 65% horas noturnas > OMS. Evitar 19–21h.","Crítico"],
            ["Nov–Dez","Transição","9–12 µg/m³","Chuvas voltam irregularmente. Filtro recomendado.","Moderado"],
        ],colWidths=[0.7*inch,1.3*inch,0.85*inch,3.2*inch,0.65*inch])
        t.setStyle(tbl_style()); story.append(t); story.append(Spacer(1,8))

        pct_v=STATIC_PCT_OMS
        if inmet is not None:
            mo=inmet.groupby("month").apply(lambda x:(x["pm25"]>15).sum()/len(x)*100).reset_index()
            mo.columns=["month","pct"]; pct_v=mo["pct"].tolist()
        fig,ax=plt.subplots(figsize=(10,3.2)); fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        bc3=[C_BLUE if i<6 else C_RED if i<10 else C_YEL for i in range(len(pct_v))]
        brs=ax.bar(MONTH_L[:len(pct_v)],pct_v,color=bc3,edgecolor="none",width=0.7)
        for b,v in zip(brs,pct_v):
            ax.text(b.get_x()+b.get_width()/2,v+0.8,f"{v:.1f}%",ha="center",va="bottom",fontsize=8,color=TXT2)
        ax.axhline(50,color=C_ORAN,lw=1.5,ls="--",label="50% das horas")
        ax.set_title("% de horas acima do limiar OMS (15 µg/m³) por mês")
        ax.set_ylabel("%",color=TXT2); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.25)
        fig.tight_layout(); add_img(fig,h=2.6*inch)

        doc.build(story)
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
