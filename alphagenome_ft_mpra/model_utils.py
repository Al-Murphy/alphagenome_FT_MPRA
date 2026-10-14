import sys
import time
from pathlib import Path

from alphagenome_research.model import dna_model

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def offline_organism_settings() -> dict:
    """Organism settings with every network-backed field disabled.

    `predict_sequence` (the only entry point this script uses) never touches
    reference FASTA, GTF annotations, splice-site tables, or variant
    calibration data -- those only support `predict_interval`/`predict_variant`.
    However, `dna_model.create(...)`'s default `organism_settings` unconditionally
    tries to fetch all of them for every organism it knows about, including a
    `gs://.../calibration_scores.pb` read that requires live GCS network
    access. On an offline HPC compute node that fetch fails with a
    FailedPreconditionError / SSL error, even though we never needed the data.

    This keeps only the small, locally-bundled output-track metadata (which
    ships inside the `alphagenome_research` package itself -- no network) and
    leaves every other field at its `None` default, so `create()` skips all
    remote reads entirely.
    """
    return {
        organism: dna_model.OrganismSettings()
        for organism in dna_model.default_organism_settings()
    }


def load_model(
    source: str,
    model_version: str,
    device_arg: str | None,
    local_checkpoint_dir: str | None = None,
) -> dna_model.AlphaGenomeModel:
    """Loads the base AlphaGenome model.

    If `source == 'local'`, loads directly from `local_checkpoint_dir` via
    `dna_model.create(...)`, bypassing kagglehub/huggingface_hub entirely --
    no login, no network access needed. Use `download_alphagenome_weights.py`
    once (on a machine with internet access) to produce that directory, then
    copy it to wherever you're running this (e.g. an HPC compute node with no
    outbound internet).
    """
    import jax

    device = None
    if device_arg is not None:
        if device_arg == "cpu":
            device = jax.devices("cpu")[0]
        elif device_arg in ("gpu", "tpu"):
            matches = [d for d in jax.devices() if d.platform == device_arg]
            if not matches:
                raise RuntimeError(f"No {device_arg} devices visible to JAX.")
            device = matches[0]
        else:
            raise ValueError(f"Unrecognized --device {device_arg!r}")
    else:
        # Auto: prefer GPU/TPU, else fall back to CPU (AlphaGenomeModel.__init__
        # raises if it finds neither a GPU/TPU nor an explicit device, so we
        # pass CPU explicitly rather than let that happen).
        accel = [d for d in jax.devices() if d.platform in ("gpu", "tpu")]
        if accel:
            device = accel[0]
        else:
            print(
                "WARNING: no GPU/TPU visible to JAX -- running on CPU. This will"
                " be slow for anything beyond a quick smoke test. AlphaGenome"
                " recommends an NVIDIA H100 GPU (see README).",
                file=sys.stderr,
            )
            device = jax.devices("cpu")[0]

    organism_settings = offline_organism_settings()

    if source == "local":
        if not local_checkpoint_dir:
            raise ValueError("--local_checkpoint_dir is required when --source local")
        # Orbax requires an absolute checkpoint path -- resolve relative paths
        # (e.g. './weights') and '~' here so the caller doesn't have to.
        resolved_dir = str(Path(local_checkpoint_dir).expanduser().resolve())
        if not Path(resolved_dir).is_dir():
            raise FileNotFoundError(
                f"--local_checkpoint_dir {local_checkpoint_dir!r} (resolved to"
                f" {resolved_dir!r}) does not exist or is not a directory."
            )
        print(
            f"Loading AlphaGenome from local checkpoint: {resolved_dir}"
            " (no Kaggle/Hugging Face login, no remote genome/calibration"
            " fetches -- fully offline)"
        )
        t0 = time.time()
        model = dna_model.create(
            resolved_dir, organism_settings=organism_settings, device=device
        )
        print(f"  loaded in {time.time() - t0:.1f}s")
        return model

    print(f"Loading AlphaGenome ({model_version}) from {source}...")
    t0 = time.time()
    if source == "kaggle":
        model = dna_model.create_from_kaggle(
            model_version, organism_settings=organism_settings, device=device
        )
    elif source == "huggingface":
        model = dna_model.create_from_huggingface(
            model_version, organism_settings=organism_settings, device=device
        )
    else:
        raise ValueError(f"Unrecognized --source {source!r}")
    print(f"  loaded in {time.time() - t0:.1f}s")
    return model