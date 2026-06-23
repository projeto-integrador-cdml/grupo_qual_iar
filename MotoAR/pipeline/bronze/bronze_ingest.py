"""
╔══════════════════════════════════════════════════════════════╗
║  MotoAR — CAMADA BRONZE (RAW)                               ║
║  Ingestão sem esquema: copia os dados brutos para a zona    ║
║  Bronze do Data Lake, adicionando metadados de ingestão.    ║
║                                                             ║
║  Uso:                                                       ║
║    python bronze_ingest.py                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ROOT aponta para pipeline/ (onde os dados e outputs estão)
ROOT    = Path(__file__).parent.parent
BRONZE  = ROOT / "data" / "bronze"

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ingest_file(src: Path, fonte: str, schema_enforcement: bool = False) -> dict:
    """
    Copia arquivo bruto para Bronze com particionamento por data e fonte.
    Schema enforcement desativado por padrão (camada Bronze = dados brutos).
    """
    if not src.exists():
        print(f"  ⚠️  Arquivo não encontrado: {src}")
        return {}

    now   = datetime.now()
    date_partition = now.strftime("%Y-%m-%d")
    dest_dir  = BRONZE / f"fonte={fonte}" / f"data={date_partition}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / src.name

    shutil.copy2(src, dest_file)
    size_kb = dest_file.stat().st_size / 1024

    meta = {
        "fonte":              fonte,
        "arquivo_origem":     str(src),
        "arquivo_bronze":     str(dest_file),
        "data_ingestao":      now.isoformat(),
        "particao_data":      date_partition,
        "tamanho_kb":         round(size_kb, 1),
        "md5":                md5(dest_file),
        "schema_enforcement": schema_enforcement,
        "camada":             "bronze",
    }

    # Salva metadados junto com o arquivo
    meta_path = dest_file.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {fonte:<20} → {dest_file.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    return meta


def main():
    parser = argparse.ArgumentParser(description="MotoAR — Bronze Ingest")
    parser.add_argument("--iqair",  default="iqair_data.csv",
                        help="Caminho para iqair_data.csv")
    parser.add_argument("--inmet",  default="ESTACOES AUTOMATICAS _ DADOS BRUTO 2025.xlsx",
                        help="Caminho para INMET xlsx")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  CAMADA BRONZE — Ingestão sem esquema")
    print("═"*60)
    print(f"  Destino: {BRONZE.relative_to(ROOT)}/")
    print(f"  Schema enforcement: OFF (dados brutos preservados)\n")

    report = {
        "camada":          "bronze",
        "run_at":          datetime.now().isoformat(),
        "schema_enforcement": False,
        "arquivos":        [],
    }

    for path_str, fonte in [
        (args.iqair,  "iqair"),
        (args.inmet,  "inmet"),
    ]:
        src = ROOT / path_str
        meta = ingest_file(src, fonte)
        if meta:
            report["arquivos"].append(meta)

    # Relatório da camada Bronze
    report_path = BRONZE / "bronze_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  📋 Relatório: {report_path.relative_to(ROOT)}")
    print(f"  ✅ Bronze concluído — {len(report['arquivos'])} arquivo(s) ingerido(s)")
    print()


if __name__ == "__main__":
    main()
