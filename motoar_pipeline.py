"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — Pipeline de Limpeza & Validação de Dados          ║
║  Etapa 02 do LCA                                            ║
║                                                             ║
║  Entradas:                                                  ║
║    • iqair_data.csv                                         ║
║    • ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx           ║
║                                                             ║
║  Saídas:                                                    ║
║    • iqair_clean.parquet / .csv                             ║
║    • inmet_clean.parquet / .csv                             ║
║    • pipeline_report.json                                   ║
║                                                             ║
║  Uso:                                                       ║
║    python motoar_pipeline.py                                ║
║    python motoar_pipeline.py --fmt csv                      ║
║    python motoar_pipeline.py --iqair outro.csv              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO CENTRAL
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # --- IQAir ---
    "iqair": {
        "status_col":       "aqi_category",
        "invalid_values":   ["não encontrado"],
        "numeric_cols": {
            "temperature": "°C",
            "humidity":    "%",
            "wind_speed":  "km/h",
        },
        "aqi_range":    (0, 500),
        "pm25_range":   (0, 500),
        "temp_range":   (-10, 60),
        "humidity_range": (0, 100),
        "wind_range":   (0, 200),
        "datetime_col": "created_at",
        "sensor_col":   "sensor_location",
        "required_cols": ["aqi", "pm25", "sensor_location", "created_at"],
    },

    # --- INMET ---
    "inmet": {
        "skiprows":      2,
        "datetime_col":  "Data/Hora",
        "status_suffix": "Status",
        "invalid_flags": ["InVld", "Inv", "INVÁLIDO"],
        "sensor_fercal": {
            "pm25":  "PM25 (ug/m3)",
            "pm10":  "PM10 (ug/m3)",
            "pts":   "PTS (ug/m3)",
            "wind_dir":   "DirVento (Deg)",
            "wind_speed": "VelVento (m/s)",
            "temp":  "TempAr (C°)",
            "rh":    "RH (%)",
            "press": "Press_Ar (mb)",
            "rad":   "RadSol (watt/m2)",
            "rain":  "Rain (mm)",
            "so2":   "SO2_ug/m3 (ug/m3)",
            "no":    "NO_ug/m3 (ug/m3)",
            "no2":   "NO2_ug/m3 (ug/m3)",
            "nox":   "NOx_ug/m3 (ug/m3)",
            "o3":    "O3_ug/m3 (ug/m3)",
            "co":    "CO_ppm (ppm)",
        },
        "sensor_escola": {
            "pm25":       "PM25 (ug/m3).1",
            "pm10":       "PM10 (ug/m3).1",
            "wind_dir":   "DirVento (Deg).1",
            "wind_speed": "VelVento (m/s).1",
            "temp":       "TempAr (C°).1",
            "rh":         "RH (%).1",
            "press":      "Press_Ar (mb).1",
            "rad":        "Radiação (watt/m2)",
            "rain":       "Rain (mm).1",
        },
        # Limites físicos plausíveis (Brasília)
        "clip_rules": {
            "pm25":       (0,   500),
            "pm10":       (0,   600),
            "pts":        (0,  1000),
            "temp":       (5,    45),
            "rh":         (0,   100),
            "press":      (850, 950),
            "wind_speed": (0,    30),
            "rain":       (0,   150),
            "so2":        (0,  1000),
            "no":         (0,   500),
            "no2":        (0,   300),
            "nox":        (0,   600),
            "o3":         (0,   500),
            "co":         (0,    20),
            "rad":        (0,  1500),
        },
    },
}

SEASON_MAP = {
    1:"chuva", 2:"chuva", 3:"chuva",
    4:"chuva", 5:"chuva", 6:"chuva",
    7:"seca",  8:"seca",  9:"seca", 10:"seca",
    11:"transição", 12:"transição",
}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────────────────

class PipelineReport:
    """Coleta e imprime métricas de cada etapa."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.steps: list[dict] = []
        self.t0 = datetime.now()

    def log(self, step: str, before: int, after: int, detail: str = ""):
        removed = before - after
        pct     = removed / before * 100 if before > 0 else 0
        entry = {
            "step":    step,
            "before":  before,
            "after":   after,
            "removed": removed,
            "pct_removed": round(pct, 2),
            "detail":  detail,
        }
        self.steps.append(entry)
        tag = f"  ✓ {step:<40} {before:>7,} → {after:>7,}  (−{removed:,} / {pct:.1f}%)"
        if detail:
            tag += f"  [{detail}]"
        print(tag)

    def summary(self) -> dict:
        elapsed = (datetime.now() - self.t0).total_seconds()
        return {
            "dataset":  self.dataset,
            "run_at":   self.t0.isoformat(),
            "elapsed_s": round(elapsed, 2),
            "steps":    self.steps,
            "total_removed": sum(s["removed"] for s in self.steps),
        }


def strip_unit(series: pd.Series, unit: str) -> pd.Series:
    """Remove sufixo de unidade e converte para float."""
    return pd.to_numeric(
        series.astype(str).str.replace(unit, "", regex=False).str.strip(),
        errors="coerce",
    )


def add_temporal_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Adiciona colunas de tempo derivadas."""
    dt = df[dt_col]
    df["year"]      = dt.dt.year
    df["month"]     = dt.dt.month
    df["day"]       = dt.dt.day
    df["hour"]      = dt.dt.hour
    df["weekday"]   = dt.dt.weekday          # 0=seg
    df["is_weekend"]= (df["weekday"] >= 5).astype(int)
    df["season"]    = df["month"].map(SEASON_MAP)
    # Codificação cíclica para ML
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_dry_season"] = df["month"].between(7, 10).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, col: str,
                          windows: list[int] = [3, 6, 24]) -> pd.DataFrame:
    """Adiciona médias móveis e lags — importante para o modelo."""
    for w in windows:
        df[f"{col}_roll{w}h"] = df[col].rolling(w, min_periods=1).mean().round(3)
    df[f"{col}_lag1h"] = df[col].shift(1)
    df[f"{col}_lag3h"] = df[col].shift(3)
    df[f"{col}_delta1h"] = df[col].diff(1).round(3)   # variação na última hora
    return df


def detect_sensor_frozen(series: pd.Series, window: int = 6) -> pd.Series:
    """Sinaliza quando o sensor repete o mesmo valor por ≥ window horas (sensor travado)."""
    rolling_std = series.rolling(window, min_periods=window).std()
    return (rolling_std == 0).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE IQAIR
# ──────────────────────────────────────────────────────────────────────────────

def clean_iqair(path: str) -> tuple[pd.DataFrame, dict]:
    print("\n" + "═"*60)
    print("  IQAIR — Limpeza & Validação")
    print("═"*60)

    rpt = PipelineReport("iqair")
    cfg = CFG["iqair"]

    # ── 1. LEITURA ────────────────────────────────────────────────────────────
    raw = pd.read_csv(path)
    n0  = len(raw)
    print(f"\n  Arquivo : {path}")
    print(f"  Shape   : {raw.shape}")
    print(f"  Colunas : {raw.columns.tolist()}\n")
    print("  ETAPAS:")

    # ── 2. COLUNAS OBRIGATÓRIAS ───────────────────────────────────────────────
    missing_cols = [c for c in cfg["required_cols"] if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")

    df = raw.copy()

    # ── 3. DATETIME ───────────────────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    n_bad_dt = df["created_at"].isna().sum()
    df = df.dropna(subset=["created_at"])
    rpt.log("Drop datetime inválido", n0, len(df), f"{n_bad_dt} timestamps nulos")

    # ── 4. STATUS INVÁLIDO ────────────────────────────────────────────────────
    if cfg["status_col"] in df.columns:
        mask_invalid = df[cfg["status_col"]].isin(cfg["invalid_values"])
        n_before = len(df)
        df = df[~mask_invalid]
        rpt.log("Drop status inválido (aqi_category)", n_before, len(df),
                f"valores: {cfg['invalid_values']}")

    # ── 5. PARSE DE UNIDADES ──────────────────────────────────────────────────
    for col, unit in cfg["numeric_cols"].items():
        if col in df.columns:
            df[col] = strip_unit(df[col], unit)

    # ── 6. RANGE CHECKS ──────────────────────────────────────────────────────
    rules = {
        "aqi":        cfg["aqi_range"],
        "pm25":       cfg["pm25_range"],
        "temperature":cfg["temp_range"],
        "humidity":   cfg["humidity_range"],
        "wind_speed": cfg["wind_range"],
    }
    for col, (lo, hi) in rules.items():
        if col in df.columns:
            n_before = len(df)
            bad = (df[col] < lo) | (df[col] > hi)
            df.loc[bad, col] = np.nan
            rpt.log(f"Nullify out-of-range: {col}", n_before, len(df),
                    f"range [{lo}, {hi}] — {bad.sum()} valores → NaN")

    # ── 7. DUPLICATAS ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["created_at", "sensor_location"])
    rpt.log("Drop duplicatas", n_before, len(df))

    # ── 8. SENSOR TRAVADO ─────────────────────────────────────────────────────
    df = df.sort_values(["sensor_location", "created_at"]).reset_index(drop=True)
    df["sensor_frozen_aqi"] = (
        df.groupby("sensor_location")["aqi"]
        .transform(lambda s: detect_sensor_frozen(s, window=6))
    )
    n_frozen = df["sensor_frozen_aqi"].sum()
    rpt.log("Detecção sensor travado (flag)", len(df), len(df),
            f"{n_frozen} registros marcados com sensor_frozen=1")

    # ── 9. FEATURES TEMPORAIS ─────────────────────────────────────────────────
    df = add_temporal_features(df, "created_at")

    # ── 10. ROLLING FEATURES ─────────────────────────────────────────────────
    df = df.sort_values(["sensor_location", "created_at"]).reset_index(drop=True)
    parts = []
    for _sensor, grp in df.groupby("sensor_location", sort=False):
        parts.append(add_rolling_features(grp.copy(), "aqi", windows=[3, 6]))
    df = pd.concat(parts, ignore_index=True)

    # ── 11. ORDENAR & RESETAR ─────────────────────────────────────────────────
    df = df.sort_values(["sensor_location", "created_at"]).reset_index(drop=True)

    # ── 12. RELATÓRIO FINAL ───────────────────────────────────────────────────
    pct_kept = len(df) / n0 * 100
    print(f"\n  RESULTADO: {n0:,} → {len(df):,} registros  ({pct_kept:.1f}% aproveitado)")
    print(f"  Colunas finais: {df.columns.tolist()}")

    return df, rpt.summary()


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE INMET
# ──────────────────────────────────────────────────────────────────────────────

def _parse_inmet_station(raw: pd.DataFrame, dt_col: str,
                         col_map: dict, station_name: str,
                         rpt: PipelineReport) -> pd.DataFrame:
    """Processa uma estação individual do arquivo INMET."""
    cfg = CFG["inmet"]
    n0  = len(raw)

    rows = []
    for _, row in raw.iterrows():
        record = {"dt": row[dt_col], "station": station_name}
        for alias, raw_col in col_map.items():
            if raw_col not in raw.columns:
                continue
            status_col = f"Status {raw_col.split(' ')[0]}"
            # tenta o status pelo sufixo padrão do INMET
            status = None
            for sc in raw.columns:
                if "Status" in sc and raw_col.replace("Status ","").split(" ")[0] in sc:
                    pass  # lookup complexo — usamos o approach abaixo
            # approach direto: pegar coluna Status imediatamente após
            col_idx = raw.columns.get_loc(raw_col) if raw_col in raw.columns else None
            if col_idx is not None and col_idx + 1 < len(raw.columns):
                next_col = raw.columns[col_idx + 1]
                if "Status" in next_col or "status" in next_col.lower():
                    status = row[next_col]
            val = row[raw_col]
            # sinaliza inválido como NaN
            if status in cfg["invalid_flags"]:
                val = np.nan
            record[alias] = val
        rows.append(record)

    df = pd.DataFrame(rows)

    # Converte tudo para numérico
    for alias in col_map.keys():
        if alias in df.columns and alias != "dt":
            df[alias] = pd.to_numeric(df[alias], errors="coerce")

    # Clipping físico
    for col, (lo, hi) in cfg["clip_rules"].items():
        if col in df.columns:
            n_clip = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            if n_clip > 0:
                rpt.log(f"Clip {station_name}.{col}", len(df), len(df),
                        f"{n_clip} valores → [{lo},{hi}]")

    return df


def clean_inmet(path: str) -> tuple[pd.DataFrame, dict]:
    print("\n" + "═"*60)
    print("  INMET — Limpeza & Validação")
    print("═"*60)

    rpt = PipelineReport("inmet")
    cfg = CFG["inmet"]

    # ── 1. LEITURA ────────────────────────────────────────────────────────────
    raw = pd.read_excel(path, skiprows=cfg["skiprows"], header=0)
    n0  = len(raw)
    print(f"\n  Arquivo : {path}")
    print(f"  Shape   : {raw.shape}")
    print("\n  ETAPAS:")

    # ── 2. DATETIME ───────────────────────────────────────────────────────────
    dt_col = cfg["datetime_col"]
    raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce")
    n_bad = raw[dt_col].isna().sum()
    raw   = raw.dropna(subset=[dt_col])
    rpt.log("Drop datetime inválido", n0, len(raw), f"{n_bad} timestamps nulos")

    # ── 3. STATUS PM25 — FERCAL ───────────────────────────────────────────────
    pm25_status_col = "Status PM25"
    if pm25_status_col in raw.columns:
        n_before = len(raw)
        raw_fercal_valid = raw[raw[pm25_status_col] == "Ok"]
        rpt.log("Filtro Status PM25 == Ok (Fercal)", n_before, len(raw_fercal_valid),
                f"{n_before - len(raw_fercal_valid)} registros inválidos")
    else:
        raw_fercal_valid = raw.copy()

    # ── 4. PROCESSAR ESTAÇÃO FERCAL ───────────────────────────────────────────
    fercal = _parse_inmet_station(
        raw_fercal_valid, dt_col,
        cfg["sensor_fercal"], "CRAS Fercal", rpt
    )

    # ── 5. PROCESSAR ESTAÇÃO ESCOLA ───────────────────────────────────────────
    # Escola não tem filtro de status PM25 próprio — usa todos os registros
    escola = _parse_inmet_station(
        raw, dt_col,
        cfg["sensor_escola"], "Escola", rpt
    )

    # ── 6. CONCAT ─────────────────────────────────────────────────────────────
    df = pd.concat([fercal, escola], ignore_index=True)
    rpt.log("Concat Fercal + Escola", len(fercal)+len(escola), len(df))

    # ── 7. SENSOR TRAVADO ─────────────────────────────────────────────────────
    df = df.sort_values(["station", "dt"]).reset_index(drop=True)
    df["sensor_frozen_pm25"] = (
        df.groupby("station")["pm25"]
        .transform(lambda s: detect_sensor_frozen(s, window=6))
    )
    n_fr = df["sensor_frozen_pm25"].sum()
    rpt.log("Detecção sensor travado (flag)", len(df), len(df),
            f"{n_fr} registros marcados com sensor_frozen=1")

    # ── 8. DUPLICATAS ─────────────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["dt", "station"])
    rpt.log("Drop duplicatas", n_before, len(df))

    # ── 9. FEATURES TEMPORAIS ─────────────────────────────────────────────────
    df = df.rename(columns={"dt": "datetime"})
    df = add_temporal_features(df, "datetime")

    # ── 10. ROLLING FEATURES ──────────────────────────────────────────────────
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    parts2 = []
    for _st, grp in df.groupby("station", sort=False):
        parts2.append(add_rolling_features(grp.copy(), "pm25", windows=[3, 6, 24]))
    df = pd.concat(parts2, ignore_index=True)

    # Chuva acumulada 6h (preditor importante de lavagem atmosférica)
    df["rain_acc6h"] = (
        df.groupby("station")["rain"]
        .transform(lambda s: s.rolling(6, min_periods=1).sum())
    )

    # Delta temperatura (proxy de inversão térmica)
    df["temp_delta1h"] = (
        df.groupby("station")["temp"]
        .transform(lambda s: s.diff(1))
    )

    # ── 11. QUALIDADE GLOBAL ──────────────────────────────────────────────────
    total = len(df)
    null_pct = df[["pm25","temp","rh","rain","no2","co"]].isnull().mean() * 100
    print("\n  % de nulos por coluna (Fercal+Escola):")
    for c, v in null_pct.items():
        bar = "█" * int(v / 5)
        print(f"    {c:<12} {v:5.1f}%  {bar}")

    pct_kept = total / (n0 * 2) * 100
    print(f"\n  RESULTADO: {n0:,}×2 → {total:,} registros  ({pct_kept:.1f}% aproveitado)")
    print(f"  Colunas finais: {df.columns.tolist()}")

    return df, rpt.summary()


# ──────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO CRUZADA BÁSICA
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate(df_inmet: pd.DataFrame, df_iq: pd.DataFrame) -> dict:
    """
    Verifica coerência entre as duas fontes.
    Ambas devem apresentar o mesmo padrão horário bimodal (pico manhã + noite).
    """
    print("\n" + "═"*60)
    print("  VALIDAÇÃO CRUZADA — INMET × IQAir")
    print("═"*60)

    checks = {}

    # Padrão horário: hora de mínimo e máximo devem coincidir
    inmet_hourly = df_inmet[df_inmet["station"]=="CRAS Fercal"].groupby("hour")["pm25"].mean()
    iq_hourly    = df_iq.groupby("hour")["aqi"].mean()

    inmet_peak  = int(inmet_hourly.idxmax())
    iq_peak     = int(iq_hourly.idxmax())
    inmet_low   = int(inmet_hourly.idxmin())
    iq_low      = int(iq_hourly.idxmin())

    print(f"\n  Pico máximo  → INMET: {inmet_peak}h  |  IQAir: {iq_peak}h")
    print(f"  Pico mínimo  → INMET: {inmet_low}h   |  IQAir: {iq_low}h")

    peak_ok = abs(inmet_peak - iq_peak) <= 3
    low_ok  = abs(inmet_low  - iq_low)  <= 3
    print(f"  Concordância pico:   {'✅ OK' if peak_ok else '⚠️  DIVERGÊNCIA'}")
    print(f"  Concordância mínimo: {'✅ OK' if low_ok else '⚠️  DIVERGÊNCIA'}")

    # Range de valores
    inmet_mean = df_inmet[df_inmet["station"]=="CRAS Fercal"]["pm25"].mean()
    iq_mean    = df_iq["aqi"].mean()
    print(f"\n  PM2.5 médio INMET: {inmet_mean:.2f} µg/m³")
    print(f"  AQI médio IQAir:   {iq_mean:.2f}  (escala diferente — comparação qualitativa)")

    checks = {
        "inmet_peak_hour": inmet_peak,
        "iqair_peak_hour": iq_peak,
        "inmet_low_hour":  inmet_low,
        "iqair_low_hour":  iq_low,
        "peak_concordant": peak_ok,
        "low_concordant":  low_ok,
        "inmet_pm25_mean": round(inmet_mean, 2),
        "iqair_aqi_mean":  round(iq_mean, 2),
    }
    return checks


# ──────────────────────────────────────────────────────────────────────────────
# SALVAR
# ──────────────────────────────────────────────────────────────────────────────

def save_output(df: pd.DataFrame, name: str, fmt: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
    else:
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
    size_kb = p.stat().st_size / 1024
    print(f"  💾 Salvo: {p}  ({size_kb:.0f} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MotoAR — Pipeline de Limpeza")
    parser.add_argument("--iqair",  default="iqair_data.csv",
                        help="Caminho para iqair_data.csv")
    parser.add_argument("--inmet",  default="ESTACOES_AUTOMATICAS___DADOS_BRUTO_2025.xlsx",
                        help="Caminho para INMET xlsx")
    parser.add_argument("--fmt",    default="parquet", choices=["parquet","csv"],
                        help="Formato de saída (parquet ou csv)")
    parser.add_argument("--out",    default="data/clean",
                        help="Diretório de saída")
    parser.add_argument("--skip-cross", action="store_true",
                        help="Pular validação cruzada")
    args = parser.parse_args()

    out_dir = Path(args.out)
    report  = {"run_at": datetime.now().isoformat(), "outputs": {}}

    # ── IQAir ────────────────────────────────────────────────────────────────
    if Path(args.iqair).exists():
        df_iq, rpt_iq = clean_iqair(args.iqair)
        save_output(df_iq, "iqair_clean", args.fmt, out_dir)
        report["iqair"] = rpt_iq
        report["outputs"]["iqair"] = str(out_dir / f"iqair_clean.{args.fmt}")
    else:
        print(f"\n⚠️  IQAir não encontrado: {args.iqair}")
        df_iq = None

    # ── INMET ────────────────────────────────────────────────────────────────
    if Path(args.inmet).exists():
        df_inmet, rpt_inmet = clean_inmet(args.inmet)
        save_output(df_inmet, "inmet_clean", args.fmt, out_dir)
        report["inmet"] = rpt_inmet
        report["outputs"]["inmet"] = str(out_dir / f"inmet_clean.{args.fmt}")
    else:
        print(f"\n⚠️  INMET não encontrado: {args.inmet}")
        df_inmet = None

    # ── VALIDAÇÃO CRUZADA ────────────────────────────────────────────────────
    if not args.skip_cross and df_iq is not None and df_inmet is not None:
        cross = cross_validate(df_inmet, df_iq)
        report["cross_validation"] = cross

    # ── RELATÓRIO JSON ───────────────────────────────────────────────────────
    report_path = out_dir / "pipeline_report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "═"*60)
    print("  PIPELINE CONCLUÍDO")
    print("═"*60)
    print(f"  Relatório : {report_path}")
    print(f"  Saídas    : {list(report.get('outputs',{}).values())}")
    print()


if __name__ == "__main__":
    main()
