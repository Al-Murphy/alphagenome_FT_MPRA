"""
Generate benchmark job scripts for hyperparameter sweeps.

This script generates 3 SLURM job scripts (one per cell type) that run
all hyperparameter combinations sequentially.
"""

from pathlib import Path

# Sequential script template (runs all combinations for one cell type)
SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ft_AG_{cell_type}_benchmark
#SBATCH --output=out/ft_AG_{cell_type}_benchmark_%j.out
#SBATCH --error=out/ft_AG_{cell_type}_benchmark_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --partition=gpuq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amurphy@cshl.edu
#SBATCH --export=ALL

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Running benchmark for cell type: {cell_type}"
echo "Total hyperparameter combinations: {num_combinations}"
echo ""

# Navigate to your project directory
cd /grid/koo/home/amurphy/projects/alphagenome_FT_MPRA

source .venv/bin/activate

# Counter for tracking progress
run_num=1
total_runs={num_combinations}
failed_runs=0
skipped_runs=0

# Function to check if a run has already completed successfully
check_run_completed() {{
    local run_name="$1"
    local results_file="./results/benchmark_results_{cell_type}.csv"
    
    if [ ! -f "$results_file" ]; then
        return 1  # File doesn't exist, run not completed
    fi
    
    # Check if run_name exists in the CSV file (first column)
    if grep -q "^$run_name," "$results_file" 2>/dev/null; then
        return 0  # Run found, already completed
    else
        return 1  # Run not found, needs to run
    fi
}}

# Function to run a single hyperparameter combination
run_experiment() {{
    local config_name="$1"
    shift
    local args="$@"
    local run_name="benchmark-{cell_type}-$config_name"
    
    echo "=========================================="
    echo "Run $run_num/$total_runs: $config_name"
    echo "=========================================="
    
    # Check if this run has already completed successfully
    if check_run_completed "$run_name"; then
        echo "⏭️  Run $run_num/$total_runs ($config_name) already completed - skipping"
        skipped_runs=$((skipped_runs + 1))
        run_num=$((run_num + 1))
        echo ""
        return 0
    fi
    
    echo "Command: python scripts/finetune_mpra.py --cell_type {cell_type} --wandb_name $run_name --use_cached_embeddings --cache_file ./.cache/embeddings/{cell_type}_train_embeddings.pkl --save_test_results ./results/benchmark_results_{cell_type}.csv $args"
    echo ""
    
    if python scripts/finetune_mpra.py \\
        --cell_type {cell_type} \\
        --wandb_name $run_name \\
        --use_cached_embeddings \\
        --cache_file ./.cache/embeddings/{cell_type}_train_embeddings.pkl \\
        --save_test_results ./results/benchmark_results_{cell_type}.csv \\
        $args; then
        echo "✓ Run $run_num/$total_runs ($config_name) completed successfully"
    else
        echo "✗ Run $run_num/$total_runs ($config_name) FAILED"
        failed_runs=$((failed_runs + 1))
    fi
    
    run_num=$((run_num + 1))
    echo ""
}}

{experiment_commands}

# Final summary
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
echo "Total runs: $total_runs"
echo "Skipped (already completed): $skipped_runs"
echo "Successful: $((total_runs - failed_runs - skipped_runs))"
echo "Failed: $failed_runs"
echo "End time: $(date)"
"""

# Define all hyperparameter combinations to test
# Note: Defaults are: sum pooling, Adam, ReLU, no dropout, nl_size=1024, no weight decay, no LR scheduler

HYPERPARAMETER_COMBINATIONS = [
    # 0. Baseline/default configuration (for comparison)
    {'job_suffix': 'baseline-default'},
    
    # 1. Pooling variations (center, flatten)
    {'pooling_type': 'center', 'job_suffix': 'pool-center'},
    {'pooling_type': 'flatten', 'job_suffix': 'pool-flatten'},
    
    # 2. Optimizer: AdamW
    {'optimizer': 'adamw', 'job_suffix': 'opt-adamw'},
    
    # 3. Activation: GELU
    {'activation': 'gelu', 'job_suffix': 'act-gelu'},
    
    # 4. Dropout variations
    {'do': 0.1, 'job_suffix': 'do-0.1'},
    {'do': 0.2, 'job_suffix': 'do-0.2'},
    {'do': 0.3, 'job_suffix': 'do-0.3'},
    {'do': 0.4, 'job_suffix': 'do-0.4'},
    {'do': 0.5, 'job_suffix': 'do-0.5'},
    
    # 5. Weight decay (L2 regularization)
    {'optimizer': 'adamw', 'weight_decay': 1e-6, 'job_suffix': 'wd-1e6'},
    
    # 6. Hidden layer configurations
    {'nl_size': '512,512', 'job_suffix': 'nl-512-512'},
    {'nl_size': '256,256', 'job_suffix': 'nl-256-256'},
    {'nl_size': '512,256', 'job_suffix': 'nl-512-256'},
    {'nl_size': '512', 'job_suffix': 'nl-512'},
    {'nl_size': '256', 'job_suffix': 'nl-256'},
    {'nl_size': '128', 'job_suffix': 'nl-128'},
    
    # 7. Learning rate schedulers
    {'lr_scheduler': 'plateau', 'job_suffix': 'lr-plateau'},
    {'lr_scheduler': 'cosine', 'no_val_split': True, 'job_suffix': 'lr-cosine'},
]

def build_experiment_args(config: dict) -> str:
    """Build command-line arguments string for a hyperparameter configuration.
    
    For baseline/default config, returns empty string (no args = all defaults).
    """
    args = []
    
    # Skip if this is the baseline (no hyperparameters specified)
    if 'job_suffix' in config and config['job_suffix'] == 'baseline-default':
        # Check if any hyperparameters are specified (they shouldn't be for baseline)
        if not any(k in config for k in ['pooling_type', 'optimizer', 'activation', 'do', 
                                         'weight_decay', 'nl_size', 'lr_scheduler', 'no_val_split']):
            return ''  # Baseline uses all defaults, no args needed
    
    if 'pooling_type' in config:
        args.append(f"--pooling_type {config['pooling_type']}")
    if 'optimizer' in config:
        args.append(f"--optimizer {config['optimizer']}")
    if 'activation' in config:
        args.append(f"--activation {config['activation']}")
    if 'do' in config:
        args.append(f"--do {config['do']}")
    if 'weight_decay' in config:
        args.append(f"--weight_decay {config['weight_decay']}")
    if 'nl_size' in config:
        args.append(f"--nl_size {config['nl_size']}")
    if 'lr_scheduler' in config:
        args.append(f"--lr_scheduler {config['lr_scheduler']}")
    if config.get('no_val_split', False):
        args.append("--no_val_split")
    
    return ' '.join(args)

def generate_sequential_script(cell_type: str, output_dir: Path):
    """Generate a sequential job script that runs all hyperparameter combinations for a cell type."""
    experiment_commands = []
    
    for config in HYPERPARAMETER_COMBINATIONS:
        # Create a copy to avoid modifying the original
        config_copy = config.copy()
        job_suffix = config_copy.pop('job_suffix')
        args_str = build_experiment_args(config_copy)
        
        experiment_commands.append(f'run_experiment "{job_suffix}" {args_str}')
    
    experiment_commands_str = '\n'.join(experiment_commands)
    
    script_content = SCRIPT_TEMPLATE.format(
        cell_type=cell_type,
        num_combinations=len(HYPERPARAMETER_COMBINATIONS),
        experiment_commands=experiment_commands_str
    )
    
    # Write script
    script_file = output_dir / f"ft_mpra_AG_{cell_type}_benchmark.sh"
    script_file.write_text(script_content)
    script_file.chmod(0o755)  # Make executable
    
    return script_file

def main():
    """Generate sequential job scripts (one per cell type)."""
    output_dir = Path('/grid/koo/home/amurphy/projects/job_scripts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cell_types = ['HepG2', 'K562', 'WTC11']
    
    for cell_type in cell_types:
        script_file = generate_sequential_script(cell_type, output_dir)
        print(f"Generated: {script_file.name}")
    
    print(f"\n✓ Generated {len(cell_types)} sequential job scripts")
    print(f"  Each script runs {len(HYPERPARAMETER_COMBINATIONS)} hyperparameter combinations")
    print(f"  Output directory: {output_dir}")

if __name__ == '__main__':
    main()
