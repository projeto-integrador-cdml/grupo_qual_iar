"""
Pre-processa os datasets reais (IQAir + INMET) e gera um único arquivo
data.json com agregações leves, prontas para serem embutidas no HTML.

Saída: data_export.json
"""
import pandas as pd
import numpy as np
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data_export.json")

MONTHS = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
SEASON_MAP = {1:"Chuva",2:"Chuva",3:"Chuva",4:"Chuva",5:"Chuva",6:"Chuva",
              7:"Seca",8:"Seca",9:"Seca",10:"Seca",
              11:"Transição",12:"Transição"}

result = {}

# ---------- IQAir ----------
iq_path = os.path.join(ROOT, "iqair_data.csv")
if os.path.exists(iq_path):
    iq = pd.read_csv(iq_path)
    iq["created_at"] = pd.to_datetime(iq["created_at"], errors="coerce")
    iq["temperature"] = pd.to_numeric(iq["temperature"].astype(str).str.replace("°C","").str.strip(), errors="coerce")
    iq["humidity"]    = pd.to_numeric(iq["humidity"].astype(str).str.replace("%","").str.strip(), errors="coerce")
    iq["wind_speed"]  = pd.to_numeric(iq["wind_speed"].astype(str).str.replace("km/h","").str.strip(), errors="coerce")
    iq["hour"] = iq["created_at"].dt.hour
    iq["date"] = iq["created_at"].dt.date

    iq_summary = {
        "total": int(len(iq)),
        "aqi_mean": float(iq["aqi"].mean()),
        "aqi_max": float(iq["aqi"].max()),
        "aqi_p50": float(iq["aqi"].median()),
        "pm25_mean": float(iq["pm25"].mean()),
        "temp_mean": float(iq["temperature"].mean()),
        "rh_mean": float(iq["humidity"].mean()),
        "sensors": sorted(iq["sensor_location"].dropna().unique().tolist()),
    }

    sensor_stats = (iq.groupby("sensor_location")
                    .agg(records=("aqi","count"),
                         aqi_mean=("aqi","mean"),
                         aqi_max=("aqi","max"),
                         aqi_std=("aqi","std"),
                         pm25_mean=("pm25","mean"),
                         temp_mean=("temperature","mean"),
                         rh_mean=("humidity","mean"))
                    .round(2).reset_index())
    iq_summary["sensor_stats"] = sensor_stats.to_dict(orient="records")

    # padrão horário por sensor
    h = (iq.groupby(["hour","sensor_location"])["aqi"].mean().round(2)
            .reset_index())
    iq_summary["hourly_by_sensor"] = {
        s: h[h["sensor_location"]==s].sort_values("hour")["aqi"].tolist()
        for s in iq_summary["sensors"]
    }

    # diário por sensor
    daily = (iq.groupby(["date","sensor_location"])["aqi"].mean().round(2)
                .reset_index())
    daily["date"] = daily["date"].astype(str)
    iq_summary["daily_by_sensor"] = {
        s: {"dates":  daily[daily["sensor_location"]==s].sort_values("date")["date"].tolist(),
            "aqi":    daily[daily["sensor_location"]==s].sort_values("date")["aqi"].tolist()}
        for s in iq_summary["sensors"]
    }

    # box plot data: percentis por sensor
    box = {}
    for s in iq_summary["sensors"]:
        v = iq[iq["sensor_location"]==s]["aqi"].dropna()
        box[s] = {
            "min": float(v.min()), "q1": float(v.quantile(.25)),
            "median": float(v.median()), "q3": float(v.quantile(.75)),
            "max": float(v.max()), "mean": float(v.mean()),
        }
    iq_summary["aqi_box_by_sensor"] = box

    result["iqair"] = iq_summary
    print(f"✓ IQAir processado: {len(iq):,} registros")
else:
    print("✗ IQAir CSV não encontrado")

# ---------- INMET ----------
xlsx_candidates = [
    os.path.join(ROOT, "ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx"),
    os.path.join(ROOT, "ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx"),
]
inmet_path = next((p for p in xlsx_candidates if os.path.exists(p)), None)

if inmet_path:
    raw = pd.read_excel(inmet_path, skiprows=2, header=0)
    raw = raw.rename(columns={"Data/Hora": "dt"})
    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["hour"] = raw["dt"].dt.hour
    raw["month"] = raw["dt"].dt.month
    raw["date"] = raw["dt"].dt.date

    valid = raw[raw["Status PM25"] == "Ok"].copy()
    valid["pm25"] = pd.to_numeric(valid["PM25 (ug/m3)"], errors="coerce")
    valid["pm10"] = pd.to_numeric(valid["PM10 (ug/m3)"], errors="coerce")
    valid["no2"]  = pd.to_numeric(valid["NO2_ug/m3 (ug/m3)"], errors="coerce").clip(0,300)
    valid["co"]   = pd.to_numeric(valid["CO_ppm (ppm)"], errors="coerce").clip(0,20)
    valid["o3"]   = pd.to_numeric(valid["O3_ug/m3 (ug/m3)"], errors="coerce").clip(0,500)
    valid["so2"]  = pd.to_numeric(valid["SO2_ug/m3 (ug/m3)"], errors="coerce").clip(0,500)
    valid["rain"] = pd.to_numeric(valid["Rain (mm)"], errors="coerce").clip(lower=0)
    valid = valid[(valid["pm25"]>=0)&(valid["pm25"]<500)]
    valid["season"] = valid["month"].map(SEASON_MAP)

    inm = {
        "total": int(len(valid)),
        "pm25_mean": float(valid["pm25"].mean()),
        "pm25_max": float(valid["pm25"].max()),
        "pm25_p50": float(valid["pm25"].median()),
        "pm25_std": float(valid["pm25"].std()),
    }

    # mensal: média, p75, max
    m = (valid.groupby("month")["pm25"]
          .agg(["mean","median","max",lambda x:x.quantile(.75)])
          .round(2).reset_index())
    m.columns = ["month","mean","median","max","p75"]
    inm["monthly"] = {
        "labels": [MONTHS[i-1] for i in m["month"]],
        "mean":   m["mean"].tolist(),
        "p75":    m["p75"].tolist(),
        "max":    m["max"].tolist(),
    }

    # padrão horário por estação
    seasons = ["Chuva","Seca","Transição"]
    hourly = {}
    for s in seasons:
        sub = valid[valid["season"]==s]
        if len(sub):
            hourly[s] = (sub.groupby("hour")["pm25"].mean().round(2)
                            .reindex(range(24), fill_value=0).tolist())
    inm["hourly_by_season"] = hourly

    # heatmap hora × mês
    heat = (valid.groupby(["hour","month"])["pm25"].mean().round(2)
                  .unstack().reindex(range(24)).reindex(columns=range(1,13)))
    inm["heatmap"] = {
        "x_months": MONTHS,
        "y_hours":  list(range(24)),
        "z":        heat.fillna(0).values.tolist(),
    }

    # histograma de PM2.5 por estação (bins)
    bins = np.linspace(0, 100, 41)
    centers = ((bins[:-1]+bins[1:])/2).round(1).tolist()
    histos = {}
    for s in seasons:
        sub = valid[(valid["season"]==s) & (valid["pm25"]<100)]
        if len(sub):
            counts, _ = np.histogram(sub["pm25"], bins=bins)
            histos[s] = counts.tolist()
    inm["histogram"] = {"bins": centers, "data": histos}

    # correlação INMET
    corr_cols = ["pm25","no2","co","o3","rain"]
    cm = valid[corr_cols].corr().round(3)
    inm["correlation"] = {
        "labels": ["PM2.5","NO2","CO","O3","Chuva"],
        "matrix": cm.values.tolist(),
    }

    # efeito chuva por categoria
    daily = (valid.groupby("date")
                  .agg(pm25=("pm25","mean"), rain=("rain","sum"))
                  .reset_index())
    bins_r = [-1, 0, 5, 20, 10000]
    labels_r = ["0mm (seco)","1–5mm (leve)","5–20mm (mod.)","≥20mm (forte)"]
    daily["rain_cat"] = pd.cut(daily["rain"], bins=bins_r, labels=labels_r)
    rain_eff = (daily.groupby("rain_cat", observed=True)["pm25"]
                       .mean().round(2).reset_index())
    inm["rain_effect"] = {
        "labels": rain_eff["rain_cat"].astype(str).tolist(),
        "pm25":   rain_eff["pm25"].tolist(),
    }

    # scatter pm25 × no2 (amostragem)
    sample = valid[valid["pm25"]<100].sample(min(800,len(valid)), random_state=42)
    inm["scatter_pm25_no2"] = {
        "no2":    sample["no2"].round(2).tolist(),
        "pm25":   sample["pm25"].round(2).tolist(),
        "season": sample["season"].tolist(),
    }

    # estatísticas descritivas para tabela
    cols_stat = ["pm25","no2","co","o3","so2","rain"]
    stats = valid[cols_stat].describe().round(2)
    inm["stats_table"] = {
        "rows":  ["PM2.5 µg/m³","NO2 µg/m³","CO ppm","O3 µg/m³","SO2 µg/m³","Chuva mm"],
        "cols":  list(stats.index),
        "data":  stats.T.values.tolist(),
    }

    # comparação INMET (PM2.5) vs IQAir (AQI) — padrão horário
    if "iqair" in result:
        inm_h = valid.groupby("hour")["pm25"].mean().round(2)
        inm["hourly_pm25_overall"] = inm_h.reindex(range(24), fill_value=0).tolist()

    # % horas que ultrapassam OMS (15 µg/m³) por mês
    pct_oms = (valid.groupby("month")["pm25"]
                    .apply(lambda x: float((x>15).mean()*100)).round(2))
    inm["pct_above_oms_15"] = pct_oms.reindex(range(1,13)).fillna(0).tolist()

    result["inmet"] = inm
    print(f"✓ INMET processado: {len(valid):,} registros")
else:
    print("✗ INMET XLSX não encontrado")

# ---------- comparação INMET × IQAir hora a hora ----------
if "inmet" in result and "iqair" in result:
    iq_h = (iq.groupby("hour")["aqi"].mean()
              .reindex(range(24), fill_value=0).round(2).tolist())
    result["compare_hourly"] = {
        "hours":  list(range(24)),
        "inmet_pm25": result["inmet"]["hourly_pm25_overall"],
        "iqair_aqi":  iq_h,
    }

# ---------- modelo (treina rápido pra extrair feature importance e métricas) ----------
if "inmet" in result:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        try:
            from xgboost import XGBRegressor as Reg
            model_name = "XGBoost"
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor as Reg
            model_name = "GradientBoosting"
        df = valid.copy().sort_values("dt").reset_index(drop=True)
        df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
        df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
        df["is_dry"]    = df["month"].between(7,10).astype(int)
        df["pm25_lag1"] = df["pm25"].shift(1).fillna(df["pm25"].mean())
        df["pm25_lag3"] = df["pm25"].shift(3).fillna(df["pm25"].mean())
        df["pm25_roll3"]= df["pm25"].rolling(3, min_periods=1).mean()
        df["rain_acc6"] = df["rain"].rolling(6, min_periods=1).sum()
        feats = ["hour_sin","hour_cos","month_sin","month_cos","is_dry",
                 "pm25_lag1","pm25_lag3","pm25_roll3","rain_acc6","no2","co"]
        df_ml = df[feats+["pm25"]].dropna()
        X, y = df_ml[feats], df_ml["pm25"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        try:
            m = Reg(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
        except TypeError:
            m = Reg(n_estimators=200, max_depth=5, random_state=42)
        m.fit(Xtr, ytr)
        ypred = m.predict(Xte)
        result["model"] = {
            "name": model_name,
            "mae":  float(round(mean_absolute_error(yte, ypred), 3)),
            "r2":   float(round(r2_score(yte, ypred), 3)),
            "n_train": int(len(Xtr)),
            "n_test":  int(len(Xte)),
            "features": feats,
            "feature_importance": [float(x) for x in m.feature_importances_],
            "sample_real":     yte.values[:120].round(2).tolist(),
            "sample_predicted": ypred[:120].round(2).tolist(),
        }
        print(f"✓ Modelo treinado: MAE={result['model']['mae']}, R²={result['model']['r2']}")
    except Exception as e:
        print(f"✗ Modelo não treinado: {e}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, separators=(",",":"))
size_kb = os.path.getsize(OUT)/1024
print(f"\n→ {OUT}  ({size_kb:.1f} KB)")
