"""Finetune AlphaGenome with MPRA head on episomal MPRA (Gosai et al. 2024).

The Gosai dataset contains ~800K 200bp sequences across 3 cell types (K562,
HepG2, SK-N-SH) measured by an episomal MPRA. Splits are chromosome-based:
test = chr7 + chr13, val = chr19 + chr21 + chrX, train = remainder.

USAGE:
    python scripts/finetune_episomal_mpra.py --cell_type K562 \\
        --config configs/episomal_K562.json
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from alphagenome.models import dna_output
from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_custom_heads,
)
from alphagenome_ft_mpra import (
    EncoderMPRAHead,
    EpisomalMPRADataset,
    MPRADataLoader,
    train,
)


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune AlphaGenome with MPRA head on episomal MPRA (Gosai 2024)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config (CLI args override config values).")

    # Data
    parser.add_argument("--cell_type", type=str, default="K562",
                        choices=["K562", "HepG2", "SKNSH"])
    parser.add_argument("--data_path", type=str, default="./data/gosai_episomal")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_shift", action="store_true", default=True)
    parser.add_argument("--random_shift_likelihood", type=float, default=0.5)
    parser.add_argument("--max_shift", type=int, default=10)
    parser.add_argument("--reverse_complement", action="store_true", default=True)
    parser.add_argument("--pad_n_bases", type=int, default=0)

    # Model
    parser.add_argument("--init_seq_len", type=int, default=200,
                        help="Episomal MPRA sequence length (Gosai = 200bp).")
    parser.add_argument("--center_bp", type=int, default=256)
    parser.add_argument("--pooling_type", type=str, default="flatten",
                        choices=["mean", "sum", "max", "center", "flatten"])
    parser.add_argument("--nl_size", type=str, default="512,512")
    parser.add_argument("--do", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu"])
    parser.add_argument("--no_freeze_backbone", action="store_true")
    parser.add_argument("--base_checkpoint_path", type=str, default=None)

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["plateau", "cosine"])
    parser.add_argument("--no_val_split", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--val_eval_frequency", type=int, default=4)
    parser.add_argument("--test_eval_frequency", type=int, default=4)

    # Two-stage
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=50)
    parser.add_argument("--resume_from_stage2", action="store_true")

    # Cached embeddings
    parser.add_argument("--use_cached_embeddings", action="store_true")
    parser.add_argument("--cache_file", type=str, default=None)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./results/models/checkpoints/episomal/")
    parser.add_argument("--save_full_model", action="store_true")
    parser.add_argument("--save_minimal_model", action="store_true")
    parser.add_argument("--no-save_minimal_model", dest="save_minimal_model",
                        action="store_false")
    parser.add_argument("--save_test_results", type=str, default=None)
    parser.add_argument("--save_val_results", type=str, default=None)

    # WandB
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="alphagenome-episomal-mpra")
    parser.add_argument("--wandb_name", type=str, default=None)

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Two-pass: read config first, then apply as defaults so CLI overrides config.
    temp_args, _ = parser.parse_known_args()
    config = None
    if temp_args.config:
        print(f"Loading config from: {temp_args.config}")
        config = load_config(temp_args.config)
        print("✓ Config loaded")

        if "cell_type" in config:
            parser.set_defaults(cell_type=config["cell_type"])
        if "data_path" in config:
            parser.set_defaults(data_path=config["data_path"])
        if "data" in config:
            d = config["data"]
            parser.set_defaults(
                batch_size=d.get("batch_size", 64),
                random_shift=d.get("random_shift", True),
                random_shift_likelihood=d.get("random_shift_likelihood", 0.5),
                max_shift=d.get("max_shift", 10),
                reverse_complement=d.get("reverse_complement", True),
                pad_n_bases=d.get("pad_n_bases", 0),
            )
        if "model" in config:
            m = config["model"]
            parser.set_defaults(
                init_seq_len=m.get("init_seq_len", 200),
                center_bp=m.get("center_bp", 256),
                pooling_type=m.get("pooling_type", "flatten"),
                nl_size=m.get("nl_size", "512,512"),
                do=m.get("do", 0.1),
                activation=m.get("activation", "relu"),
            )
        if "training" in config:
            t = config["training"]
            parser.set_defaults(
                num_epochs=t.get("num_epochs", 100),
                learning_rate=t.get("learning_rate", 1e-3),
                optimizer=t.get("optimizer", "adam"),
                weight_decay=t.get("weight_decay", 1e-6),
                gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
                gradient_clip=t.get("gradient_clip", None),
                lr_scheduler=t.get("lr_scheduler", None),
                no_val_split=t.get("no_val_split", False),
                early_stopping_patience=t.get("early_stopping_patience", 5),
                val_eval_frequency=t.get("val_eval_frequency", 4),
                test_eval_frequency=t.get("test_eval_frequency", 4),
            )
        if "two_stage" in config and config["two_stage"].get("enabled", False):
            ts = config["two_stage"]
            parser.set_defaults(
                second_stage_lr=ts.get("second_stage_lr", None),
                second_stage_epochs=ts.get("second_stage_epochs", 50),
            )
        if "cached_embeddings" in config:
            ce = config["cached_embeddings"]
            parser.set_defaults(
                use_cached_embeddings=ce.get("use_cached_embeddings", False),
                cache_file=ce.get("cache_file", None),
            )
        if "checkpointing" in config:
            cp = config["checkpointing"]
            parser.set_defaults(
                checkpoint_dir=cp.get(
                    "checkpoint_dir",
                    "./results/models/checkpoints/episomal/",
                ),
            )
        if "wandb" in config:
            wb = config["wandb"]
            parser.set_defaults(
                no_wandb=not wb.get("enabled", True),
                wandb_project=wb.get("project", "alphagenome-episomal-mpra"),
                wandb_name=wb.get("wandb_name", None),
            )
        if "base_checkpoint_path" in config:
            parser.set_defaults(base_checkpoint_path=config["base_checkpoint_path"])

    args = parser.parse_args()

    # Default save mode: minimal unless explicitly set.
    save_full_set = "--save_full_model" in sys.argv
    no_save_minimal_set = "--no-save_minimal_model" in sys.argv
    if save_full_set or no_save_minimal_set:
        args.save_minimal_model = False
    else:
        args.save_minimal_model = True

    # Default WandB run name
    if args.wandb_name is None:
        args.wandb_name = f"episomal-{args.cell_type}-seed{args.seed}"

    checkpoint_path = (
        Path(args.checkpoint_dir) / args.cell_type / args.wandb_name
    ).resolve()

    print("=" * 80)
    print("AlphaGenome Episomal MPRA Fine-tuning (Gosai 2024)")
    print("=" * 80)
    print(f"Cell type:                  {args.cell_type}")
    print(f"Data path:                  {args.data_path}")
    print(f"Batch size:                 {args.batch_size}")
    print(f"Init seq len:               {args.init_seq_len}")
    print(f"Pooling type:               {args.pooling_type}")
    print(f"nl_size / do:               {args.nl_size} / {args.do}")
    print(f"Learning rate:              {args.learning_rate}")
    print(f"Optimizer / WD:             {args.optimizer} / {args.weight_decay}")
    print(f"Num epochs:                 {args.num_epochs}")
    print(f"Freeze backbone:            {not args.no_freeze_backbone}")
    if args.second_stage_lr:
        print(f"Two-stage:                  Stage1={args.num_epochs}ep, "
              f"Stage2={args.second_stage_epochs}ep @ lr={args.second_stage_lr}")
    print(f"Checkpoint path:            {checkpoint_path}")
    print(f"Seed:                       {args.seed}")
    print("=" * 80)

    # Parse nl_size
    if "," in args.nl_size:
        nl_size = [int(x.strip()) for x in args.nl_size.split(",")]
    else:
        nl_size = int(args.nl_size)

    # Register custom MPRA head
    print("\nRegistering custom MPRA head...")
    register_custom_head(
        "mpra_head",
        EncoderMPRAHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name="mpra_head",
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={
                "center_bp": args.center_bp,
                "pooling_type": args.pooling_type,
                "nl_size": nl_size,
                "do": args.do,
                "activation": args.activation,
            },
        ),
    )
    print("✓ Custom head registered")

    # Determine init_seq_len (encoder-output expects concrete length)
    init_seq_len = args.init_seq_len
    if args.use_cached_embeddings:
        if args.cache_file is None:
            raise ValueError("--cache_file must be provided when --use_cached_embeddings")
        import pickle
        with open(args.cache_file, "rb") as f:
            cache_data = pickle.load(f)
        sample = next(iter(cache_data.values()))
        init_seq_len = sample.shape[0] * 128
        print(f"Inferred init_seq_len from cache: {init_seq_len}")
        del cache_data
    else:
        if args.pad_n_bases > 0:
            init_seq_len += args.pad_n_bases
        print(f"Using sequence length: {init_seq_len} bp")

    # Create model
    print("\nCreating model with custom heads...")
    model_with_custom = create_model_with_custom_heads(
        "all_folds",
        custom_heads=["mpra_head"],
        checkpoint_path=args.base_checkpoint_path,
        use_encoder_output=True,
        init_seq_len=init_seq_len,
    )
    print("✓ Model created")

    if not args.no_freeze_backbone:
        model_with_custom.freeze_except_head("mpra_head")
        print("✓ Backbone frozen (Stage 1)")

    # Cached embeddings can't combine with two-stage
    if args.use_cached_embeddings and args.second_stage_lr is not None:
        raise ValueError("Cached embeddings incompatible with two-stage training.")

    # Build datasets
    print(f"\nLoading episomal datasets (cell_type={args.cell_type})...")
    rng_train = jax.random.PRNGKey(args.seed)
    rng_val = jax.random.PRNGKey(args.seed + 1)
    rng_test = jax.random.PRNGKey(args.seed + 2)

    train_dataset = EpisomalMPRADataset(
        model=model_with_custom,
        path_to_data=args.data_path,
        cell_type=args.cell_type,
        split="train",
        random_shift=args.random_shift if not args.use_cached_embeddings else False,
        random_shift_likelihood=args.random_shift_likelihood,
        max_shift=args.max_shift,
        reverse_complement=args.reverse_complement if not args.use_cached_embeddings else False,
        pad_n_bases=args.pad_n_bases,
        rng_key=rng_train,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=args.cache_file if args.use_cached_embeddings else None,
    )

    val_cache = test_cache = None
    if args.use_cached_embeddings and args.cache_file:
        cache_path = Path(args.cache_file)
        val_cache = str(cache_path.parent / f"{args.cell_type}_val_embeddings.pkl")
        test_cache = str(cache_path.parent / f"{args.cell_type}_test_embeddings.pkl")

    if args.no_val_split:
        val_dataset = None
    else:
        val_dataset = EpisomalMPRADataset(
            model=model_with_custom,
            path_to_data=args.data_path,
            cell_type=args.cell_type,
            split="val",
            random_shift=False,
            reverse_complement=False,
            pad_n_bases=args.pad_n_bases,
            rng_key=rng_val,
            use_cached_embeddings=args.use_cached_embeddings,
            cache_file=val_cache,
        )

    test_dataset = EpisomalMPRADataset(
        model=model_with_custom,
        path_to_data=args.data_path,
        cell_type=args.cell_type,
        split="test",
        random_shift=False,
        reverse_complement=False,
        pad_n_bases=args.pad_n_bases,
        rng_key=rng_test,
        use_cached_embeddings=args.use_cached_embeddings,
        cache_file=test_cache,
    )

    # Dataloaders
    train_loader = MPRADataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (MPRADataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
                  if val_dataset else None)
    test_loader = MPRADataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"✓ Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset) if val_dataset else 0} | "
          f"Test: {len(test_dataset)}")

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    wandb_config = {
        "cell_type": args.cell_type,
        "data_source": "gosai_episomal",
        "init_seq_len": args.init_seq_len,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "pooling_type": args.pooling_type,
        "nl_size": args.nl_size,
        "activation": args.activation,
        "dropout": args.do,
        "lr_scheduler": args.lr_scheduler,
        "no_val_split": args.no_val_split,
        "seed": args.seed,
    }

    history = train(
        model_with_custom,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=str(checkpoint_path),
        save_full_model=args.save_full_model,
        save_minimal_model=args.save_minimal_model,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip=args.gradient_clip,
        val_eval_frequency=args.val_eval_frequency,
        test_eval_frequency=args.test_eval_frequency,
        second_stage_lr=args.second_stage_lr,
        second_stage_epochs=args.second_stage_epochs,
        resume_from_stage2=args.resume_from_stage2,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_config=wandb_config,
        use_cached_embeddings=args.use_cached_embeddings,
        lr_scheduler=args.lr_scheduler,
        save_test_results=args.save_test_results,
        save_val_results=args.save_val_results,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    if history.get("val_pearson"):
        print(f"Best val Pearson:  {max(history['val_pearson']):.4f}")
    if history.get("test_pearson"):
        print(f"Final test Pearson: {history['test_pearson'][-1]:.4f}")
    return history


if __name__ == "__main__":
    main()
