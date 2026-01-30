# Configuration Files

This directory contains JSON configuration files with optimal hyperparameters for each dataset and cell type.

## Available Config Files

### LentiMPRA Cell Lines
- `mpra_HepG2.json` - Optimal hyperparameters for HepG2 cell line
- `mpra_K562.json` - Optimal hyperparameters for K562 cell line  
- `mpra_WTC11.json` - Optimal hyperparameters for WTC11 cell line

### DeepSTARR
- `starrseq.json` - Optimal hyperparameters for DeepSTARR dataset

## Usage

### Basic Usage

Load a config file when running finetuning:

```bash
# LentiMPRA with config
python scripts/finetune_mpra.py --config configs/mpra_HepG2.json

# DeepSTARR with config
python scripts/finetune_starrseq.py --config configs/starrseq.json
```

### Override Config Values

Command-line arguments override config file values:

```bash
# Use config but override learning rate
python scripts/finetune_mpra.py \
    --config configs/mpra_HepG2.json \
    --learning_rate 2e-3

# Use config but change batch size
python scripts/finetune_starrseq.py \
    --config configs/starrseq.json \
    --batch_size 64
```

### Populating Config Files

After running benchmarks, update the config files with optimal hyperparameters:

1. Review benchmark results:
   ```bash
   python scripts/collate_benchmark_results.py --results_dir results/
   ```

2. Identify best hyperparameters from the summary tables

3. Edit the appropriate config file (e.g., `configs/mpra_HepG2.json`) and update values:
   ```json
   {
     "model": {
       "pooling_type": "sum",  // Update with best pooling type
       "nl_size": "1024",       // Update with best hidden layer size
       "do": 0.2,              // Update with best dropout
       "activation": "gelu"     // Update with best activation
     },
     "training": {
       "learning_rate": 0.001,  // Update with best learning rate
       "optimizer": "adamw",     // Update with best optimizer
       "weight_decay": 1e-5      // Update with best weight decay
     }
   }
   ```

4. For two-stage training, enable it and set parameters:
   ```json
   {
     "two_stage": {
       "enabled": true,
       "second_stage_lr": 1e-5,
       "second_stage_epochs": 50
     },
     "training": {
       "num_epochs": 3  // Stage 1 epochs (when to switch to Stage 2)
     }
   }
   ```

## Config File Structure

Each config file contains the following sections:

- **`data`**: Data loading parameters (batch size, augmentations, etc.)
- **`model`**: Model architecture parameters (pooling, hidden layers, dropout, etc.)
- **`training`**: Training hyperparameters (learning rate, optimizer, epochs, etc.)
- **`two_stage`**: Two-stage training configuration (if enabled)
- **`cached_embeddings`**: Cached embeddings settings (for fast Stage 1 training)
- **`checkpointing`**: Checkpoint directory and save settings
- **`wandb`**: Weights & Biases logging configuration
- **`base_checkpoint_path`**: Optional local AlphaGenome checkpoint path

## Example: Complete Workflow

1. **Run benchmarks** to find optimal hyperparameters:
   ```bash
   sbatch /path/to/ft_mpra_AG_HepG2_benchmark_stage1.sh
   sbatch /path/to/ft_mpra_AG_HepG2_benchmark_stage2.sh
   ```

2. **Collate results**:
   ```bash
   python scripts/collate_benchmark_results.py
   ```

3. **Update config** with best hyperparameters:
   ```bash
   # Edit configs/mpra_HepG2.json with optimal values
   nano configs/mpra_HepG2.json
   ```

4. **Train final model** with optimal config:
   ```bash
   python scripts/finetune_mpra.py \
       --config configs/mpra_HepG2.json \
       --wandb_name "mpra-HepG2-final"
   ```

## Notes

- Config files use JSON format with `null` for unset optional values
- All paths in config files are relative to the project root
- Command-line arguments always override config file values
- Config files are version-controlled, so you can track hyperparameter evolution
