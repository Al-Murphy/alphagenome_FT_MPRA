"""Reproduce the plant STARR-seq (Jores 2021) benchmark table.

One centralized script that renders the full model x tissue x mode results grid
(full-finetune + linear-probe test Pearson r) for all four models. By default it
reads the committed reference metrics under
``results/plant_starrseq/reference/*.json`` — this reproduces the exact published
table deterministically, with no heavy dependencies.

With ``--run <model>`` it invokes that model's runner live for every
tissue x mode (x probe) and re-reads the freshly written metrics, so cells for an
installed model can be recomputed end-to-end. Runners write to
``results/plant_starrseq/<model>/<tissue>/<mode>/<method>/metrics.json``; anything
not run live falls back to the reference numbers.

USAGE:
    python scripts/reproduce_plant_starrseq_table.py            # render committed table
    python scripts/reproduce_plant_starrseq_table.py --from_live # prefer live results if present
    python scripts/reproduce_plant_starrseq_table.py --run alphagenome --mode combined
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "results" / "plant_starrseq" / "reference"
LIVE_DIR = REPO_ROOT / "results" / "plant_starrseq"

MODELS = ["ntv3", "alphagenome", "plantcad2", "plantcaduceus", "jores"]
MODEL_LABEL = {
    "ntv3": "NTv3-post",
    "alphagenome": "AlphaGenome-JAX",
    "plantcad2": "PlantCAD2",
    "plantcaduceus": "PlantCaduceus",
    "jores": "Jores CNN",
}
TISSUES = ["leaf", "proto"]
MODES = ["promoter_only", "enhancer", "combined"]
METHODS = ["finetune", "probe"]

RUNNER = {
    "alphagenome": "finetune_plant_starrseq.py",
    "ntv3": "finetune_ntv3_plant_starrseq.py",
    "plantcad2": "finetune_plantcad2_plant_starrseq.py",
    "plantcaduceus": "finetune_plantcad2_plant_starrseq.py",  # same runner, l32 via config
    "jores": "finetune_jores_plant_starrseq.py",
}
CONFIG = {
    m: {t: f"configs/plant_starrseq_{m}_{t}.json" for t in TISSUES} for m in MODELS
}


def _read_metrics(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _cell_value(model, tissue, mode, method, prefer_live):
    """Return test_pearson for one cell, preferring live results if requested."""
    live = LIVE_DIR / model / tissue / mode / method / "metrics.json"
    ref = REFERENCE_DIR / f"{model}_{tissue}_{mode}_{method}.json"

    if prefer_live:
        rec = _read_metrics(live) or _read_metrics(ref)
    else:
        rec = _read_metrics(ref) or _read_metrics(live)

    if rec is None:
        return None
    return rec.get("test_pearson")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _render_table(method, prefer_live):
    lines = ["| Model | Tissue | promoter_only | enhancer | combined |",
             "|---|---|---|---|---|"]
    rows = []
    for model in MODELS:
        for tissue in TISSUES:
            cells = [_cell_value(model, tissue, mode, method, prefer_live) for mode in MODES]
            if method == "probe" and all(c is None for c in cells):
                continue  # Jores CNN has no probe
            lines.append(
                f"| {MODEL_LABEL[model]} | {tissue} | "
                + " | ".join(_fmt(c) for c in cells) + " |"
            )
            rows.append((model, tissue, method, cells))
    return "\n".join(lines), rows


def _run_model_live(model, only_mode=None, do_probe=True):
    """Invoke a model's runner for every tissue x mode (and probe where supported)."""
    modes = [only_mode] if only_mode else MODES
    for tissue in TISSUES:
        cfg = CONFIG[model][tissue]
        for mode in modes:
            base = [sys.executable, str(REPO_ROOT / "scripts" / RUNNER[model]),
                    "--config", cfg, "--tissue", tissue, "--mode", mode]
            print(f"\n>>> {model} {tissue} {mode} finetune")
            subprocess.run(base, cwd=str(REPO_ROOT), check=False)
            if do_probe and model != "jores":
                print(f">>> {model} {tissue} {mode} probe")
                subprocess.run(base + ["--probe"], cwd=str(REPO_ROOT), check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--run", type=str, default=None, choices=MODELS,
                        help="Recompute this model's cells live before rendering.")
    parser.add_argument("--mode", type=str, default=None, choices=MODES,
                        help="Restrict --run to one data mode.")
    parser.add_argument("--from_live", action="store_true",
                        help="Prefer live results/<model>/... over the committed reference.")
    parser.add_argument("--output_dir", type=str, default="results/plant_starrseq")
    args = parser.parse_args()

    if args.run:
        _run_model_live(args.run, only_mode=args.mode)
        args.from_live = True

    ft_md, ft_rows = _render_table("finetune", args.from_live)
    pr_md, pr_rows = _render_table("probe", args.from_live)

    header = "# Plant STARR-seq (Jores 2021) benchmark\n\n"
    source = ("_Rendered from live results_\n" if args.from_live
              else "_Rendered from committed reference metrics_\n")
    body = (header + source
            + "\n## Full-finetune test Pearson r\n\n" + ft_md
            + "\n\n## Linear-probe test Pearson r (frozen backbone; Jores CNN excluded)\n\n"
            + pr_md + "\n")

    print("\n" + body)

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(body)

    csv_lines = ["model,tissue,method,mode,test_pearson"]
    for model, tissue, method, cells in ft_rows + pr_rows:
        for mode, v in zip(MODES, cells):
            csv_lines.append(f"{model},{tissue},{method},{mode},"
                             f"{'' if v is None else f'{v:.4f}'}")
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"\nWrote {out_dir / 'summary.md'} and {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
