"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — ORQUESTRADOR (Airflow-style DAG)                  ║
║                                                             ║
║  Executa o pipeline ponta a ponta em ordem:                 ║
║    1. Bronze  → ingestão dos dados brutos                   ║
║    2. Silver  → limpeza e validação                         ║
║    3. Gold    → agregações + modelo + MLflow                ║
║    4. Quality → validações de qualidade                     ║
║                                                             ║
║  Uso:                                                       ║
║    python orchestration/run_pipeline.py                     ║
║    python orchestration/run_pipeline.py --skip-model        ║
║    python orchestration/run_pipeline.py --fmt csv           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    # Set UTF-8 mode for stdout
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent.parent


# ─── ESTRUTURA DE TAREFAS (simula Airflow DAG) ────────────────────────────────

class Task:
    def __init__(self, task_id: str, script: str, args: list = None):
        self.task_id   = task_id
        self.script    = ROOT / script
        self.args      = args or []
        self.status    = "pending"
        self.duration  = 0.0
        self.output    = ""

    def run(self) -> bool:
        print(f"\n  ┌─ Task: {self.task_id}")
        print(f"  │  Script: {self.script.name}")
        t0 = time.time()
        try:
            # Define ambiente com encoding UTF-8 para suportar caracteres especiais
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            result = subprocess.run(
                [sys.executable, str(self.script)] + self.args,
                capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=str(ROOT), env=env
            )
            self.duration = time.time() - t0
            self.output   = result.stdout + result.stderr

            # Imprime output da task
            for line in result.stdout.strip().splitlines():
                print(f"  │  {line}")
            if result.returncode != 0:
                for line in result.stderr.strip().splitlines():
                    print(f"  │  ❌ {line}")
                self.status = "failed"
                print(f"  └─ ❌ FALHOU ({self.duration:.1f}s)")
                return False

            self.status = "success"
            print(f"  └─ ✅ OK ({self.duration:.1f}s)")
            return True

        except Exception as e:
            self.duration = time.time() - t0
            self.status   = "error"
            self.output   = str(e)
            print(f"  └─ 💥 ERRO: {e}")
            return False


# ─── DEFINIÇÃO DO DAG ─────────────────────────────────────────────────────────

def build_dag(fmt: str, skip_model: bool, inmet_file: str) -> list[Task]:
    """Retorna a lista de tarefas na ordem de execução."""
    model_arg = ["--skip-model"] if skip_model else []

    dag = [
        Task("t1_bronze_ingest",   "bronze/bronze_ingest.py",   ["--inmet", inmet_file]),
        Task("t2_silver_transform","silver/silver_transform.py", ["--fmt", fmt]),
        Task("t3_gold_build",      "gold/gold_build.py",       ["--fmt", fmt] + model_arg),
        Task("t4_quality_check",   "quality/quality_check.py", []),
    ]
    return dag


# ─── EXECUÇÃO DO PIPELINE ─────────────────────────────────────────────────────

def run_dag(tasks: list[Task], stop_on_fail: bool = True) -> dict:
    print("\n" + "═"*60)
    print("  MotoAR PIPELINE — Orquestrador")
    print(f"  Run ID: {datetime.now().strftime('%Y%m%d-%H%M%S')}")
    print("═"*60)
    print(f"  Tasks: {' → '.join(t.task_id for t in tasks)}")

    t_start  = time.time()
    success  = 0
    failed   = 0

    for task in tasks:
        ok = task.run()
        if ok:
            success += 1
        else:
            failed += 1
            if stop_on_fail:
                print(f"\n  ⛔ Pipeline parado em '{task.task_id}' (stop_on_fail=True)")
                break

    total_time = time.time() - t_start

    # Relatório final
    print("\n" + "═"*60)
    print("  RESUMO DO PIPELINE")
    print("═"*60)
    for t in tasks:
        icon = {"success":"✅","failed":"❌","error":"💥","pending":"⏸️"}.get(t.status,"?")
        print(f"  {icon} {t.task_id:<30} {t.status:<10} {t.duration:.1f}s")
    print(f"\n  Total: {success} OK · {failed} falha(s) · {total_time:.1f}s")

    report = {
        "run_at":     datetime.now().isoformat(),
        "total_time": round(total_time, 2),
        "success":    success,
        "failed":     failed,
        "tasks": [
            {"id": t.task_id, "status": t.status, "duration": round(t.duration, 2)}
            for t in tasks
        ],
    }

    report_dir  = ROOT / "orchestration"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Mantém sempre o último relatório com nome fixo (fácil de achar)
    with open(report_dir / "last_run.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  📋 Relatório: orchestration/last_run.json")
    print()
    return report


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MotoAR — Orquestrador de Pipeline")
    parser.add_argument("--fmt",         default="parquet", choices=["parquet","csv"])
    parser.add_argument("--inmet",       default="../data/raw/ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx")
    parser.add_argument("--iqair",       default="../data/raw/iqair_data.csv")
    parser.add_argument("--skip-model",  action="store_true", help="Pula treinamento do modelo")
    parser.add_argument("--no-stop",     action="store_true", help="Continua mesmo com falha")
    args = parser.parse_args()

    # Constrói DAG com caminhos corretos
    dag = [
        Task("t1_bronze_ingest",   "bronze/bronze_ingest.py",   ["--inmet", args.inmet, "--iqair", args.iqair]),
        Task("t2_silver_transform","silver/silver_transform.py", ["--fmt", args.fmt]),
        Task("t3_gold_build",      "gold/gold_build.py",       ["--fmt", args.fmt] + (["--skip-model"] if args.skip_model else [])),
        Task("t4_quality_check",   "quality/quality_check.py", []),
    ]
    
    report = run_dag(dag, stop_on_fail=not args.no_stop)

    sys.exit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
