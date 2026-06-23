"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — VALIDAÇÃO DE QUALIDADE (Great Expectations-style) ║
║                                                             ║
║  Valida os dados Silver antes de chegar à camada Gold.     ║
║  Bloqueie anomalias, nulos excessivos e valores fora de    ║
║  range físico esperado para Brasília-DF.                   ║
║                                                             ║
║  Uso:                                                       ║
║    python quality/quality_check.py                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
SILVER = ROOT / "data" / "silver"
QUAL   = ROOT / "quality"
QUAL.mkdir(exist_ok=True)


# ─── EXPECTATIVAS (regras de qualidade) ───────────────────────────────────────

INMET_EXPECTATIONS = [
    # (nome, coluna, tipo, valor, crítico)
    ("não_nulo_pm25",      "pm25",   "not_null_pct",   0.90,  True),
    ("range_pm25",         "pm25",   "between",        (0, 500), True),
    ("range_temp",         "temp",   "between",        (5, 45),  False),
    ("range_rh",           "rh",     "between",        (0, 100), False),
    ("range_no2",          "no2",    "between",        (0, 300), False),
    ("range_co",           "co",     "between",        (0, 20),  False),
    ("nao_duplicado",      None,     "no_duplicates",  ["datetime","station"], True),
    ("frozen_sensor_pct",  "sensor_frozen", "max_mean", 0.40,   False),
    ("pm25_media_razoavel","pm25",   "mean_between",   (0, 100), True),
    ("n_minimo_registros", None,     "min_rows",       1000,     True),
]

IQAIR_EXPECTATIONS = [
    ("não_nulo_aqi",       "aqi",    "not_null_pct",   0.90,  True),
    ("range_aqi",          "aqi",    "between",        (0, 500), True),
    ("range_pm25_iq",      "pm25",   "between",        (0, 500), True),
    ("nao_duplicado",      None,     "no_duplicates",  ["created_at","sensor_location"], True),
    ("frozen_sensor_pct",  "sensor_frozen", "max_mean", 0.90,  False),
    ("n_minimo_registros", None,     "min_rows",       5000,     True),
]


def check(df: pd.DataFrame, expectations: list) -> list:
    results = []

    for exp_name, col, exp_type, value, critical in expectations:
        passed  = False
        detail  = ""

        try:
            if exp_type == "not_null_pct":
                pct = df[col].notna().mean()
                passed = pct >= value
                detail = f"{pct:.1%} não-nulos (mínimo {value:.0%})"

            elif exp_type == "between":
                lo, hi = value
                if col in df.columns:
                    sub    = df[col].dropna()
                    pct_ok = ((sub >= lo) & (sub <= hi)).mean()
                    passed = pct_ok >= 0.99
                    detail = f"{pct_ok:.1%} dentro de [{lo}, {hi}]"
                else:
                    passed = True
                    detail = "coluna ausente — ignorado"

            elif exp_type == "no_duplicates":
                cols_dup = [c for c in value if c in df.columns]
                if cols_dup:
                    n_dup  = df.duplicated(subset=cols_dup).sum()
                    passed = n_dup == 0
                    detail = f"{n_dup} duplicatas detectadas"
                else:
                    passed = True
                    detail = "colunas ausentes — ignorado"

            elif exp_type == "max_mean":
                if col in df.columns:
                    mean_v = df[col].mean()
                    passed = mean_v <= value
                    detail = f"média={mean_v:.3f} (máx={value})"
                else:
                    passed = True
                    detail = "coluna ausente — ignorado"

            elif exp_type == "mean_between":
                lo, hi = value
                if col in df.columns:
                    mean_v = df[col].mean()
                    passed = lo <= mean_v <= hi
                    detail = f"média={mean_v:.2f} (esperado [{lo},{hi}])"
                else:
                    passed = False
                    detail = "coluna ausente"

            elif exp_type == "min_rows":
                passed = len(df) >= value
                detail = f"{len(df):,} linhas (mínimo {value:,})"

        except Exception as e:
            passed = False
            detail = f"Erro: {e}"

        icon = "✅" if passed else ("❌" if critical else "⚠️")
        severity = "critical" if (not passed and critical) else \
                   "warning"  if (not passed and not critical) else "ok"

        results.append({
            "expectation": exp_name,
            "column":      col or "—",
            "passed":      bool(passed),
            "severity":    severity,
            "detail":      detail,
        })
        print(f"  {icon} {exp_name:<35} {detail}")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  VALIDAÇÃO DE QUALIDADE — Silver Layer")
    print("═"*60)

    all_results = {}
    has_critical_fail = False

    # ── INMET Silver ──
    inmet_path = SILVER / "inmet_silver.parquet"
    if not inmet_path.exists():
        inmet_path = SILVER / "inmet_silver.csv"

    if inmet_path.exists():
        print(f"\n  INMET Silver: {inmet_path.name}")
        df_inmet = (pd.read_parquet(inmet_path) if inmet_path.suffix == ".parquet"
                    else pd.read_csv(inmet_path))
        results_inmet = check(df_inmet, INMET_EXPECTATIONS)
        all_results["inmet"] = results_inmet
        fails = [r for r in results_inmet if not r["passed"] and r["severity"] == "critical"]
        if fails:
            has_critical_fail = True
    else:
        print("\n  ⚠️  INMET Silver não encontrado — execute silver_transform.py")
        all_results["inmet"] = []

    # ── IQAir Silver ──
    iqair_path = SILVER / "iqair_silver.parquet"
    if not iqair_path.exists():
        iqair_path = SILVER / "iqair_silver.csv"

    if iqair_path.exists():
        print(f"\n  IQAir Silver: {iqair_path.name}")
        df_iqair = (pd.read_parquet(iqair_path) if iqair_path.suffix == ".parquet"
                    else pd.read_csv(iqair_path))
        results_iqair = check(df_iqair, IQAIR_EXPECTATIONS)
        all_results["iqair"] = results_iqair
        fails = [r for r in results_iqair if not r["passed"] and r["severity"] == "critical"]
        if fails:
            has_critical_fail = True
    else:
        print("\n  ⚠️  IQAir Silver não encontrado — execute silver_transform.py")
        all_results["iqair"] = []

    # Resumo
    total = sum(len(v) for v in all_results.values())
    ok    = sum(1 for v in all_results.values() for r in v if r["passed"])
    warns = sum(1 for v in all_results.values() for r in v if r["severity"] == "warning")
    crits = sum(1 for v in all_results.values() for r in v if r["severity"] == "critical")

    print("\n" + "─"*60)
    print(f"  Resultado: {ok}/{total} OK · {warns} avisos · {crits} críticos")
    if has_critical_fail:
        print("  ❌ Falha crítica detectada — revisar dados antes de prosseguir")
    else:
        print("  ✅ Dados aprovados para a camada Gold")

    # Salva relatório
    report = {
        "run_at":           datetime.now().isoformat(),
        "has_critical_fail": bool(has_critical_fail),
        "summary":          {"total": total, "ok": ok, "warnings": warns, "critical": crits},
        "results":          all_results,
    }
    report_path = QUAL / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  📋 Relatório: quality/quality_report.json\n")

    # Retorna código de saída 1 se houver falha crítica
    import sys
    sys.exit(1 if has_critical_fail else 0)


if __name__ == "__main__":
    main()
