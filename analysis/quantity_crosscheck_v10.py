"""v10 §4 — Cross-table quantity consistency check.

Scans paper/sections/*.tex for numbers that should agree across tables
and emits quantity_crosscheck.csv flagging any inconsistencies.

This is the check that should have caught the BR-LZ 64µs vs 3ms split.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = json.loads((ROOT / "experiments/runs/master_results.json").read_text())


def latest(pattern):
    fps = sorted(ROOT.glob(pattern))
    return fps[-1] if fps else None


def get_unified_brlz():
    """Authoritative BR-LZ p50 from v10 unified harness."""
    fp = latest("experiments/runs/*_v10_unified/unified.json")
    if not fp: return {}
    d = json.loads(fp.read_text())
    out = {}
    for region, rd in d["regions"].items():
        for r, rrd in rd["radii"].items():
            for name, dat in rrd["indices"].items():
                if name == "BR-LZ":
                    out[f"{region}/{r}"] = dat["p50_us"]
    return out


def grep_in_tex(pattern, files):
    out = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(pattern, line):
                out.append((f.name, i, m.group(0)))
    return out


def main():
    tex_files = list((ROOT / "paper" / "sections").glob("*.tex"))
    rows = []

    # 1. BR-LZ p50 should be ~3000us everywhere it appears
    brlz_unified = get_unified_brlz()
    print("=== BR-LZ p50 unified (DMA, NOAA, Norway, Piraeus) ===")
    for k, v in brlz_unified.items():
        print(f"  {k}: {v:.1f} us")

    # find any BR-LZ p50 mention < 1000us that's NOT in build/footprint context
    susp = []
    for f in tex_files:
        text = f.read_text()
        for m in re.finditer(r"BR-?LZ[^\n]{0,40}?(\d{2,3})\s*[µu]\\?,?s", text):
            v = int(m.group(1))
            if v < 1000:
                # context — is it a p50 row?
                ctx = text[max(0, m.start()-100):m.end()+50]
                if "p50" in ctx or "$p_{50}$" in ctx or "p_{50}" in ctx:
                    susp.append((f.name, m.group(0), ctx[-150:]))

    rows.append(["BR-LZ_p50_unified_DMA_5km", brlz_unified.get("DMA/5000m"),
                 "us", "experiments/runs/*_v10_unified/unified.json", "authoritative"])
    rows.append(["BR-LZ_p50_NOAA_5km", brlz_unified.get("NOAA/5000m"),
                 "us", "experiments/runs/*_v10_unified/unified.json", "authoritative"])
    rows.append(["BR-LZ_p50_Norway_5km", brlz_unified.get("Norway/5000m"),
                 "us", "experiments/runs/*_v10_unified/unified.json", "authoritative"])
    rows.append(["BR-LZ_p50_Piraeus_5km", brlz_unified.get("Piraeus/5000m"),
                 "us", "experiments/runs/*_v10_unified/unified.json", "authoritative"])

    if susp:
        print("\n=== FLAG: suspicious low BR-LZ p50 mentions ===")
        for f, n, c in susp:
            print(f"  {f}: '{n}' (ctx: ...{c}...)")
            rows.append(["BR-LZ_p50_paper", n, "DRIFT", f, c[:80]])

    # 2. Scaffolding scan
    scaff_patterns = [
        "filled in", "filled by", "run completes", "land before submission",
        r"\\textit{\(run", "TODO", "XXX", "FIXME", "placeholder"
    ]
    print("\n=== scaffolding scan ===")
    found = False
    for p in scaff_patterns:
        hits = grep_in_tex(p, tex_files)
        for f, n, m in hits:
            print(f"  SCAFFOLD {f}:{n}: '{m}'")
            rows.append(["scaffolding", m, "DRIFT", f"{f}:{n}", "remove"])
            found = True
    if not found: print("  clean")

    # 3. Arithmetic re-derivation
    print("\n=== arithmetic checks ===")
    checks = [
        ("SDF_speedup", 176.4 / 1.08, 163.0, 0.5),
        ("OSM_libspat_warm_speedup", 13.1 / 0.0106, 1236, 50),
        ("Neighbor_bx_speedup", 88.2 / 0.0142, 6212, 50),
        ("R*_size_ratio", 3172 / 1285, 2.47, 0.1),
    ]
    for n, computed, expected, tol in checks:
        ok = abs(computed - expected) < tol
        flag = "OK" if ok else "DRIFT"
        print(f"  [{flag}] {n}: computed={computed:.2f}  expected={expected}  Δ={abs(computed-expected):.2f}")
        rows.append([n, f"{computed:.2f}", flag, "arithmetic", str(expected)])

    out_csv = ROOT / "quantity_crosscheck.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "status", "source", "notes"])
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
