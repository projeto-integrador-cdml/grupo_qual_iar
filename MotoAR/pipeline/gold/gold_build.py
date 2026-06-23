"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — CAMADA GOLD (BUSINESS READY)                     ║
║  Agregações analíticas, engenharia de features (86),       ║
║  treinamento do modelo XGBoost com rastreamento MLflow.    ║
║                                                             ║
║  Uso:                                                       ║
║    python gold_build.py                                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import pickle
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

# ROOT aponta para pipeline/ (onde os dados estão)
ROOT   = Path(__file__).parent.parent
SILVER = ROOT / "data" / "silver"
GOLD   = ROOT / "data" / "gold"

MONTHS = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
SEASON_MAP = {
    1:"Chuva", 2:"Chuva", 3:"Chuva", 4:"Chuva", 5:"Chuva", 6:"Chuva",
    7:"Seca",  8:"Seca",  9:"Seca",  10:"Seca",
    11:"Transição", 12:"Transição",
}


def load_silver(fonte: str, fmt: str) -> pd.DataFrame | None:
    path = SILVER / f"{fonte}_silver.{fmt}"
    if not path.exists():
        path = SILVER / f"{fonte}_silver.{'csv' if fmt == 'parquet' else 'parquet'}"
    if not path.exists():
        print(f"  ⚠️  Silver não encontrado para '{fonte}': {path}")
        return None
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"  ✓ Silver {fonte}: {len(df):,} registros")
    return df


def build_aggregations(df_inmet: pd.DataFrame, df_iqair: pd.DataFrame | None) -> dict:
    """Gera todas as agregações Gold (prontas para o dashboard)."""
    result = {}

    # ── IQAir ──
    if df_iqair is not None:
        iq = df_iqair.copy()
        if "created_at" in iq.columns:
            iq["created_at"] = pd.to_datetime(iq["created_at"], errors="coerce")
            iq["hour"] = iq["created_at"].dt.hour
            iq["date"] = iq["created_at"].dt.date
        sensors = sorted(iq["sensor_location"].dropna().unique().tolist())

        result["iqair"] = {
            "total":      int(len(iq)),
            "aqi_mean":   float(iq["aqi"].mean()),
            "aqi_max":    float(iq["aqi"].max()),
            "pm25_mean":  float(iq["pm25"].mean()),
            "sensors":    sensors,
            "sensor_stats": (
                iq.groupby("sensor_location")
                  .agg(records=("aqi","count"), aqi_mean=("aqi","mean"),
                       aqi_max=("aqi","max"), pm25_mean=("pm25","mean"))
                  .round(2).reset_index().to_dict(orient="records")
            ),
            "hourly_by_sensor": {
                s: iq[iq["sensor_location"]==s].groupby("hour")["aqi"]
                     .mean().round(2).reindex(range(24), fill_value=0).tolist()
                for s in sensors
            },
        }

    # ── INMET ──
    v = df_inmet.copy()
    if "datetime" in v.columns:
        v["datetime"] = pd.to_datetime(v["datetime"], errors="coerce")
        v["hour"]     = v["datetime"].dt.hour
        v["month"]    = v["datetime"].dt.month
        v["date"]     = v["datetime"].dt.date
    v["season"] = v["month"].map(SEASON_MAP)

    # Mensal
    m = v.groupby("month")["pm25"].agg(
        ["mean","median","max", lambda x: x.quantile(.75)]
    ).reset_index()
    m.columns = ["month","mean","median","max","p75"]

    # % horas acima OMS
    pct_oms = (
        v.groupby("month")["pm25"]
         .apply(lambda x: float((x > 15).mean() * 100))
         .round(2)
    )

    # Heatmap hora × mês
    heat = (
        v.groupby(["hour","month"])["pm25"].mean().round(2)
         .unstack().reindex(range(24)).reindex(columns=range(1,13))
    )

    # Efeito chuva
    daily = v.groupby("date").agg(pm25=("pm25","mean"), rain=("rain","sum")).reset_index()
    bins_r  = [-1, 0, 5, 20, 10000]
    labels_r = ["0mm (seco)","1–5mm (leve)","5–20mm (mod.)","≥20mm (forte)"]
    daily["rain_cat"] = pd.cut(daily["rain"], bins=bins_r, labels=labels_r)
    rain_eff = daily.groupby("rain_cat", observed=True)["pm25"].mean().round(2).reset_index()

    # Correlação
    corr_cols = [c for c in ["pm25","no2","co","o3","rain"] if c in v.columns]
    cm = v[corr_cols].corr().round(3)

    result["inmet"] = {
        "total":     int(len(v)),
        "pm25_mean": float(v["pm25"].mean()),
        "pm25_max":  float(v["pm25"].max()),
        "pm25_std":  float(v["pm25"].std()),
        "monthly": {
            "labels": [MONTHS[i-1] for i in m["month"]],
            "mean":   m["mean"].tolist(),
            "p75":    m["p75"].tolist(),
            "max":    m["max"].tolist(),
        },
        "hourly_by_season": {
            s: v[v["season"]==s].groupby("hour")["pm25"].mean().round(2)
                 .reindex(range(24), fill_value=0).tolist()
            for s in ["Chuva","Seca","Transição"]
            if s in v["season"].values
        },
        "heatmap": {
            "x_months": MONTHS,
            "y_hours":  list(range(24)),
            "z":        heat.fillna(0).values.tolist(),
        },
        "rain_effect": {
            "labels": rain_eff["rain_cat"].astype(str).tolist(),
            "pm25":   rain_eff["pm25"].tolist(),
        },
        "correlation": {
            "labels": [c.upper() for c in corr_cols],
            "matrix": cm.values.tolist(),
        },
        "pct_above_oms_15": pct_oms.reindex(range(1,13)).fillna(0).tolist(),
    }

    return result


def train_model_with_mlflow(df_inmet: pd.DataFrame, mlflow_uri: str) -> dict:
    """Treina o modelo XGBoost com rastreamento MLflow."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    db_path = ROOT / "mlruns" / "motoar.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_uri = f"sqlite:///{db_path.resolve().as_posix()}"
    mlflow.set_tracking_uri(sqlite_uri)
    mlflow.set_experiment("MotoAR-PM25-Predictor")

    df = df_inmet.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").reset_index(drop=True)
        if "hour"  not in df.columns: df["hour"]  = df["datetime"].dt.hour
        if "month" not in df.columns: df["month"] = df["datetime"].dt.month

    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
    df["is_dry"]     = df["month"].between(7,10).astype(int)
    df["pm25_lag1"]  = df["pm25"].shift(1).fillna(df["pm25"].mean())
    df["pm25_lag3"]  = df["pm25"].shift(3).fillna(df["pm25"].mean())
    df["pm25_roll3"] = df["pm25"].rolling(3, min_periods=1).mean()
    df["rain_acc6"]  = df["rain"].rolling(6, min_periods=1).sum() if "rain" in df.columns else 0

    features = ["hour_sin","hour_cos","month_sin","month_cos","is_dry",
                "pm25_lag1","pm25_lag3","pm25_roll3","rain_acc6"]
    for extra in ["no2","co"]:
        if extra in df.columns:
            features.append(extra)

    df_ml = df[features + ["pm25"]].dropna()
    X, y  = df_ml[features], df_ml["pm25"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": 200, "max_depth": 6,
        "learning_rate": 0.05, "random_state": 42,
    }

    try:
        from xgboost import XGBRegressor
        model_name = "XGBoost"
        model = XGBRegressor(**params, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model_name = "GradientBoosting"
        model = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )

    with mlflow.start_run(run_name=f"motoar-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        mae = float(round(mean_absolute_error(y_te, y_pred), 3))
        r2  = float(round(r2_score(y_te, y_pred), 3))

        # Registra no MLflow
        mlflow.log_params({**params, "model": model_name, "features": len(features)})
        mlflow.log_metrics({"mae": mae, "r2": r2,
                            "n_train": len(X_tr), "n_test": len(X_te)})
        mlflow.log_param("feature_names", ",".join(features))

        # Salva artefato do modelo
        model_path = GOLD / "xgb_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(model_path))

        run_id = mlflow.active_run().info.run_id

    print(f"  ✓ Modelo: {model_name}  MAE={mae}  R²={r2}")
    print(f"  ✓ MLflow run_id: {run_id}")

    return {
        "model_name": model_name,
        "mae":  mae, "r2": r2,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "features": features,
        "feature_importance": [float(x) for x in model.feature_importances_],
        "sample_real":      y_te.values[:120].round(2).tolist(),
        "sample_predicted": y_pred[:120].round(2).tolist(),
        "mlflow_run_id":    run_id,
        "model_path":       str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(description="MotoAR — Gold Build")
    parser.add_argument("--fmt",         default="parquet", choices=["parquet","csv"])
    parser.add_argument(
        "--mlflow-uri",
        default=f"sqlite:///{(ROOT / 'mlruns' / 'motoar.db').resolve().as_posix()}",
        help="URI do MLflow tracking",
    )
    parser.add_argument("--skip-model",  action="store_true",
                        help="Pula o treinamento do modelo")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  CAMADA GOLD — Agregações + Feature Engineering + Modelo")
    print("═"*60)
    print(f"  MLflow URI: {args.mlflow_uri}\n")

    GOLD.mkdir(parents=True, exist_ok=True)

    # Carrega Silver
    df_inmet = load_silver("inmet", args.fmt)
    df_iqair = load_silver("iqair", args.fmt)

    if df_inmet is None:
        print("\n  ❌ INMET Silver não encontrado. Execute silver_transform.py primeiro.")
        return

    # Agrega
    print("\n  Gerando agregações Gold...")
    result = build_aggregations(df_inmet, df_iqair)

    # Modelo com MLflow
    if not args.skip_model:
        print("\n  Treinando modelo com MLflow...")
        result["model"] = train_model_with_mlflow(df_inmet, args.mlflow_uri)

    # Salva data_export.json na raiz (consumido pelo dashboard)
    export_path = ROOT / "data_export.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",",":"))
    print(f"\n  💾 data_export.json → {export_path.relative_to(ROOT)}")

    # Salva cópia na camada Gold
    gold_export = GOLD / "gold_aggregations.json"
    with open(gold_export, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  💾 Agregações Gold → {gold_export.relative_to(ROOT)}")

    # Relatório da camada Gold
    report = {
        "camada":   "gold",
        "run_at":   datetime.now().isoformat(),
        "n_inmet":  len(df_inmet),
        "n_iqair":  len(df_iqair) if df_iqair is not None else 0,
        "model":    result.get("model", {}),
    }
    with open(GOLD / "gold_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n  ✅ Gold concluído")
    if "model" in result:
        print(f"     MAE={result['model']['mae']}  R²={result['model']['r2']}")
    print()


if __name__ == "__main__":
    main()
