"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — CAMADA SILVER (CLEAN)                             ║
║  Limpeza, deduplicação e tratamento de nulos sobre os       ║
║  dados brutos da camada Bronze.                             ║
║                                                             ║
║  Uso:                                                       ║
║    python silver_transform.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# ROOT aponta para pipeline/ (onde os dados estão)
ROOT   = Path(__file__).parent.parent
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"

SEASON_MAP = {
    1:"chuva", 2:"chuva", 3:"chuva", 4:"chuva", 5:"chuva", 6:"chuva",
    7:"seca",  8:"seca",  9:"seca",  10:"seca",
    11:"transição", 12:"transição",
}

# ─── UTILITÁRIOS ──────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    dt = df[dt_col]
    df["year"]       = dt.dt.year
    df["month"]      = dt.dt.month
    df["day"]        = dt.dt.day
    df["hour"]       = dt.dt.hour
    df["weekday"]    = dt.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["season"]     = df["month"].map(SEASON_MAP)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["is_dry_season"] = df["month"].between(7, 10).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, col: str,
                          windows: list = [3, 6, 24]) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll{w}h"] = df[col].rolling(w, min_periods=1).mean().round(3)
    df[f"{col}_lag1h"]   = df[col].shift(1)
    df[f"{col}_lag3h"]   = df[col].shift(3)
    df[f"{col}_delta1h"] = df[col].diff(1).round(3)
    return df


def detect_frozen(series: pd.Series, window: int = 6) -> pd.Series:
    return (series.rolling(window, min_periods=window).std() == 0).astype(int)


def find_latest_bronze(fonte: str) -> Path | None:
    """Encontra o arquivo mais recente na camada Bronze para uma fonte."""
    pattern = list(BRONZE.glob(f"fonte={fonte}/**/*.csv")) + \
              list(BRONZE.glob(f"fonte={fonte}/**/*.xlsx"))
    if not pattern:
        return None
    return max(pattern, key=lambda p: p.stat().st_mtime)


# ─── SILVER IQAIR ─────────────────────────────────────────────────────────────

def process_iqair_silver(path: Path, fmt: str, report: dict) -> pd.DataFrame | None:
    print("\n  ── IQAir → Silver ──")

    raw = pd.read_csv(path)
    n0  = len(raw)

    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce")
    raw = raw.dropna(subset=["created_at"])

    # Remove status inválido
    if "aqi_category" in raw.columns:
        raw = raw[~raw["aqi_category"].isin(["não encontrado"])]

    # Parse unidades
    for col, unit in [("temperature", "°C"), ("humidity", "%"), ("wind_speed", "km/h")]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(
                raw[col].astype(str).str.replace(unit, "", regex=False).str.strip(),
                errors="coerce"
            )

    # Range checks → NaN
    ranges = {"aqi": (0,500), "pm25": (0,500), "temperature": (-10,60),
              "humidity": (0,100), "wind_speed": (0,200)}
    for col, (lo, hi) in ranges.items():
        if col in raw.columns:
            raw.loc[(raw[col] < lo) | (raw[col] > hi), col] = np.nan

    # Deduplicação
    raw = raw.drop_duplicates(subset=["created_at", "sensor_location"])

    # Detecção de sensor travado
    raw = raw.sort_values(["sensor_location", "created_at"]).reset_index(drop=True)
    raw["sensor_frozen"] = (
        raw.groupby("sensor_location")["aqi"]
        .transform(lambda s: detect_frozen(s, window=6))
    )

    # Features temporais e rolling
    raw = add_temporal_features(raw, "created_at")
    parts = []
    for _, grp in raw.groupby("sensor_location", sort=False):
        parts.append(add_rolling_features(grp.copy(), "aqi", windows=[3, 6]))
    raw = pd.concat(parts).sort_values(["sensor_location", "created_at"]).reset_index(drop=True)

    # Salvar
    out = SILVER / f"iqair_silver.{fmt}"
    SILVER.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        raw.to_parquet(out, index=False)
    else:
        raw.to_csv(out, index=False)

    pct = len(raw) / n0 * 100
    print(f"    {n0:,} → {len(raw):,} registros ({pct:.1f}% aproveitado)")
    print(f"    💾 {out.relative_to(ROOT)}")

    report["iqair"] = {
        "n_raw": n0, "n_clean": len(raw),
        "pct_kept": round(pct, 1),
        "colunas": raw.columns.tolist(),
        "output": str(out),
    }
    return raw


# ─── SILVER INMET ─────────────────────────────────────────────────────────────

def process_inmet_silver(path: Path, fmt: str, report: dict) -> pd.DataFrame | None:
    print("\n  ── INMET → Silver ──")

    raw = pd.read_excel(path, skiprows=2, header=0)
    n0  = len(raw)

    raw["datetime"] = pd.to_datetime(raw["Data/Hora"], errors="coerce")
    raw = raw.dropna(subset=["datetime"])

    # Filtro qualidade PM2.5
    if "Status PM25" in raw.columns:
        valid = raw[raw["Status PM25"] == "Ok"].copy()
    else:
        valid = raw.copy()

    # Renomeia e converte colunas principais
    col_map = {
        "PM25 (ug/m3)":       "pm25",
        "PM10 (ug/m3)":       "pm10",
        "TempAr (C°)":        "temp",
        "RH (%)":             "rh",
        "Rain (mm)":          "rain",
        "NO2_ug/m3 (ug/m3)":  "no2",
        "CO_ppm (ppm)":       "co",
        "O3_ug/m3 (ug/m3)":   "o3",
        "SO2_ug/m3 (ug/m3)":  "so2",
        "VelVento (m/s)":     "wind_speed",
        "DirVento (Deg)":     "wind_dir",
    }
    for src_col, dst_col in col_map.items():
        if src_col in valid.columns:
            valid[dst_col] = pd.to_numeric(valid[src_col], errors="coerce")

    valid["station"] = "CRAS Fercal"

    # Clipping físico
    clips = {
        "pm25": (0, 500), "pm10": (0, 600), "temp": (5, 45),
        "rh": (0, 100), "rain": (0, 150), "no2": (0, 300),
        "co": (0, 20), "o3": (0, 500), "so2": (0, 1000),
        "wind_speed": (0, 30),
    }
    for col, (lo, hi) in clips.items():
        if col in valid.columns:
            valid[col] = valid[col].clip(lo, hi)

    # Deduplicação
    valid = valid.drop_duplicates(subset=["datetime", "station"])

    # Sensor travado
    valid = valid.sort_values("datetime").reset_index(drop=True)
    valid["sensor_frozen"] = detect_frozen(valid["pm25"], window=6)

    # Features temporais e rolling
    valid = add_temporal_features(valid, "datetime")
    valid = add_rolling_features(valid, "pm25", windows=[3, 6, 24])

    # Chuva acumulada 6h
    valid["rain_acc6h"] = valid["rain"].rolling(6, min_periods=1).sum()

    # Salvar
    SILVER.mkdir(parents=True, exist_ok=True)
    cols_keep = (
        ["datetime", "station", "pm25", "pm10", "temp", "rh", "rain",
         "no2", "co", "o3", "so2", "wind_speed", "wind_dir", "sensor_frozen",
         "year", "month", "day", "hour", "weekday", "is_weekend", "season",
         "hour_sin", "hour_cos", "month_sin", "month_cos", "is_dry_season",
         "pm25_roll3h", "pm25_roll6h", "pm25_roll24h",
         "pm25_lag1h", "pm25_lag3h", "pm25_delta1h", "rain_acc6h"]
    )
    cols_keep = [c for c in cols_keep if c in valid.columns]
    out_df = valid[cols_keep]

    out = SILVER / f"inmet_silver.{fmt}"
    if fmt == "parquet":
        out_df.to_parquet(out, index=False)
    else:
        out_df.to_csv(out, index=False)

    pct = len(out_df) / n0 * 100
    print(f"    {n0:,} → {len(out_df):,} registros ({pct:.1f}% aproveitado)")
    print(f"    💾 {out.relative_to(ROOT)}")

    report["inmet"] = {
        "n_raw": n0, "n_clean": len(out_df),
        "pct_kept": round(pct, 1),
        "colunas": cols_keep,
        "output": str(out),
    }
    return out_df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MotoAR — Silver Transform")
    parser.add_argument("--fmt",   default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--iqair", default=None, help="Caminho do CSV IQAir (Bronze ou raiz)")
    parser.add_argument("--inmet", default=None, help="Caminho do XLSX INMET (Bronze ou raiz)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  CAMADA SILVER — Limpeza, Deduplicação, Validação")
    print("═"*60)

    report = {"camada": "silver", "run_at": datetime.now().isoformat()}

    # Resolve caminhos: prioriza Bronze, depois raiz
    iqair_path = (
        Path(args.iqair) if args.iqair else
        find_latest_bronze("iqair") or ROOT / "iqair_data.csv"
    )
    inmet_path = (
        Path(args.inmet) if args.inmet else
        find_latest_bronze("inmet") or ROOT / "ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx"
    )

    print(f"  IQAir: {iqair_path}")
    print(f"  INMET: {inmet_path}")

    if iqair_path.exists():
        process_iqair_silver(iqair_path, args.fmt, report)
    else:
        print(f"\n  ⚠️  IQAir não encontrado: {iqair_path}")

    if inmet_path.exists():
        process_inmet_silver(inmet_path, args.fmt, report)
    else:
        print(f"\n  ⚠️  INMET não encontrado: {inmet_path}")

    report_path = SILVER / "silver_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  📋 Relatório: {report_path.relative_to(ROOT)}")
    print("  ✅ Silver concluído\n")


if __name__ == "__main__":
    main()
