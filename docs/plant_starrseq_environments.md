# Plant STARR-seq runner environments

The plant STARR-seq (Jores 2021) benchmark covers four models whose dependency
stacks **conflict** and cannot coexist in one environment. Use a separate
environment per model, each built from its own requirements file under
[`requirements/`](../requirements). The AlphaGenome path runs in the repo's
default environment; the other three each need a dedicated env.

| Model | Runner | Environment | Requirements | Why separate |
|---|---|---|---|---|
| AlphaGenome | `scripts/finetune_plant_starrseq.py` | the repo's default env (`pip install -e .`) | (repo `pyproject.toml`) | — |
| NTv3-post | `scripts/finetune_ntv3_plant_starrseq.py` | `plant_ntv3` | `requirements/plant_starrseq_ntv3.txt` | pins `jax[cuda12]==0.9.2` + **CPU** torch |
| PlantCAD2 | `scripts/finetune_plantcad2_plant_starrseq.py` | `plant_plantcad2` | `requirements/plant_starrseq_plantcad2.txt` | **CUDA-12.4** torch + `mamba-ssm`, `transformers<5` |
| Jores CNN | `scripts/finetune_jores_plant_starrseq.py` | `plant_jores` | `requirements/plant_starrseq_jores.txt` | plain torch, from scratch |

The three non-AlphaGenome envs are mutually exclusive: NTv3 needs `jax[cuda12]==0.9.2`
with a **CPU** torch wheel, PlantCAD2 needs a **cu124** torch wheel plus Mamba CUDA
kernels, and their `transformers` pins differ — so they must be installed apart.

> **Package import note.** All runners import the JAX-free helpers
> `alphagenome_ft_mpra.plant_starrseq_utils` / `alphagenome_ft_mpra.plant_torch`.
> The package's JAX/AlphaGenome exports are optional (guarded), so
> `import alphagenome_ft_mpra.plant_torch` works in these torch/JAX-only envs. Make
> the package importable in each env with `pip install -e .` (or by adding the repo
> root to `PYTHONPATH`); this does **not** pull the AlphaGenome stack.

---

## AlphaGenome (repo default env)

Runs in the environment you already installed for this repo (see the top-level
README). No extra requirements file.

```bash
python scripts/finetune_plant_starrseq.py --config configs/plant_starrseq_alphagenome_leaf.json --mode combined
python scripts/finetune_plant_starrseq.py --config configs/plant_starrseq_alphagenome_leaf.json --mode combined --probe
```

## NTv3-post (`plant_ntv3`)

JAX + Flax NNX. The backbone (`nucleotide_transformer_v3`) is installed from the
InstaDeep repo. At **run** time JAX needs its bundled libs on the loader path.

```bash
conda create -n plant_ntv3 python=3.12 -y
conda activate plant_ntv3
pip install -r requirements/plant_starrseq_ntv3.txt
pip install -e .                      # make alphagenome_ft_mpra importable (no AG stack pulled)

# JAX runtime needs the conda libstdc++/CUDA libs:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python scripts/finetune_ntv3_plant_starrseq.py --config configs/plant_starrseq_ntv3_leaf.json --mode combined
python scripts/finetune_ntv3_plant_starrseq.py --config configs/plant_starrseq_ntv3_leaf.json --mode combined --probe
```

If `torch` resolves to a CUDA wheel and clashes with the JAX CUDA runtime, force the
CPU build (the runner only uses torch for the DataLoader-free ridge helpers):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## PlantCAD2 (`plant_plantcad2`)

Caduceus / Mamba2. `mamba-ssm` reads `torch` at build time, so install it in **two
steps**: base deps (which include torch) first, then the Mamba kernels.

```bash
conda create -n plant_plantcad2 python=3.12 -y
conda activate plant_plantcad2

# 1) torch (cu124) + the rest of the base deps
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/plant_starrseq_plantcad2.txt
pip install -e .

# 2) Mamba kernels — build sees the installed torch
pip install --no-build-isolation "causal-conv1d>=1.4" "mamba-ssm>=2.0"
```

**No CUDA toolchain (no `nvcc`)?** Building `mamba-ssm` from source needs `nvcc`.
If the cluster ships only the CUDA runtime, install the upstream **prebuilt wheels**
instead (these match torch 2.6 / cu12 / cp312 — pick wheels matching your
`torch{maj.min}` if you change torch):

```bash
pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

Then:

```bash
python scripts/finetune_plantcad2_plant_starrseq.py --config configs/plant_starrseq_plantcad2_leaf.json --mode combined
python scripts/finetune_plantcad2_plant_starrseq.py --config configs/plant_starrseq_plantcad2_leaf.json --mode combined --probe
```

## Jores CNN (`plant_jores`)

Trained from scratch — any recent PyTorch works, no special kernels.

```bash
conda create -n plant_jores python=3.12 -y
conda activate plant_jores
pip install -r requirements/plant_starrseq_jores.txt
pip install -e .

python scripts/finetune_jores_plant_starrseq.py --config configs/plant_starrseq_jores_leaf.json --mode combined
```

---

## Reproducing the benchmark table

`scripts/reproduce_plant_starrseq_table.py` has **no heavy dependencies** and renders
the committed benchmark from `results/plant_starrseq/reference/`. Run it from any of
the environments above (or a bare Python with the repo importable):

```bash
python scripts/reproduce_plant_starrseq_table.py                 # committed reference table
python scripts/reproduce_plant_starrseq_table.py --run plantcad2 # recompute PlantCAD2 cells live
```
