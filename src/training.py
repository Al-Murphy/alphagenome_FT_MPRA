"""
Training loop functions for AlphaGenome MPRA finetuning.
"""

from typing import Any
import jax
import jax.numpy as jnp
from alphagenome_research.model import dna_model

try:
    import optax
except ImportError:
    print("Warning: optax not installed. Install with: pip install optax")
    optax = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from .data import MPRADataLoader


def train_epoch(
    model: Any,
    dataloader: MPRADataLoader,
    optimizer_state: Any,
    opt_update: Any,
    rng_key: jax.Array,
    loss_fn: Any,
    head_name: str = 'mpra_head',
    gradient_accumulation_steps: int = 1,
) -> tuple[dict, Any, jax.Array]:
    """Train for one epoch with optional gradient accumulation.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for training batches
        optimizer_state: Optax optimizer state
        opt_update: Optax update function
        rng_key: Random key for training
        loss_fn: Loss function created by model.create_loss_fn_for_head()
        head_name: Name of the custom head to train
        gradient_accumulation_steps: Number of mini-batches to accumulate gradients
            before updating parameters. Use > 1 to reduce memory usage.
        
    Returns:
        Tuple of (metrics_dict, updated_optimizer_state, updated_rng_key)
    """
    
    total_loss = 0.0
    total_pearson = 0.0
    num_batches = 0
    accumulated_grads = None
    accumulated_loss = 0.0
    accumulated_pearson = 0.0
    step_count = 0
    
    for batch in dataloader:
        rng_key, subkey = jax.random.split(rng_key)
        
        # Split batch into mini-batches for gradient accumulation
        batch_size = batch['seq'].shape[0]
        mini_batch_size = max(1, batch_size // gradient_accumulation_steps)
        
        for mini_batch_start in range(0, batch_size, mini_batch_size):
            mini_batch_end = min(mini_batch_start + mini_batch_size, batch_size)
            
            # Extract mini-batch
            mini_batch = {
                'seq': batch['seq'][mini_batch_start:mini_batch_end],
                'y': batch['y'][mini_batch_start:mini_batch_end],
                'organism_index': batch['organism_index'][mini_batch_start:mini_batch_end],
            }
            
            # Compute loss and gradients for mini-batch
            def loss_fn_inner(params):
                # Get predictions
                with model._device_context:
                    predictions = model._predict(
                        params,
                        model._state,
                        mini_batch['seq'],
                        mini_batch['organism_index'],
                        negative_strand_mask=jnp.zeros(len(mini_batch['seq']), dtype=bool),
                        strand_reindexing=jax.device_put(
                            model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                            model._device_context._device
                        ),
                    )
                
                # Get predictions for our head
                head_predictions = predictions[head_name]
                
                # Prepare batch for head's loss method (expects 'targets' key)
                loss_batch = {'targets': mini_batch['y']}
                
                # Use loss function to compute loss
                loss_dict = loss_fn(head_predictions, loss_batch)
                loss = loss_dict['loss']
                
                return loss, loss_dict
            
            # Compute gradients for mini-batch
            grad_fn = jax.value_and_grad(loss_fn_inner, has_aux=True)
            (loss, loss_dict), grads = grad_fn(model._params)
            
            # Extract pearson from loss_dict
            pearson = loss_dict.get('pearson_corr', 0.0)
            
            # Accumulate gradients and loss
            if accumulated_grads is None:
                # Initialize accumulated gradients with first gradient
                accumulated_grads = grads
                accumulated_loss = loss
                accumulated_pearson = pearson
            else:
                # Add gradients (they'll be averaged when we update)
                accumulated_grads = jax.tree.map(
                    lambda acc, new: acc + new,
                    accumulated_grads,
                    grads
                )
                accumulated_loss += loss
                accumulated_pearson += pearson
            
            step_count += 1
            
            # Update parameters after accumulating enough gradients
            if step_count >= gradient_accumulation_steps:
                # Average accumulated gradients
                accumulated_grads = jax.tree.map(
                    lambda g: g / gradient_accumulation_steps,
                    accumulated_grads
                )
                
                # Update parameters
                updates, optimizer_state = opt_update(accumulated_grads, optimizer_state, model._params)
                model._params = optax.apply_updates(model._params, updates)
                
                # Track average loss and pearson
                avg_loss = accumulated_loss / gradient_accumulation_steps
                avg_pearson = accumulated_pearson / gradient_accumulation_steps
                total_loss += float(avg_loss)
                total_pearson += float(avg_pearson)
                num_batches += 1
                
                # Reset accumulation
                accumulated_grads = None
                accumulated_loss = 0.0
                accumulated_pearson = 0.0
                step_count = 0
        
        # Handle remaining accumulated gradients if batch doesn't divide evenly
        if step_count > 0:
            # Average accumulated gradients
            accumulated_grads = jax.tree.map(
                lambda g: g / step_count,
                accumulated_grads
            )
            
            # Update parameters
            updates, optimizer_state = opt_update(accumulated_grads, optimizer_state, model._params)
            model._params = optax.apply_updates(model._params, updates)
            
            # Track average loss and pearson
            avg_loss = accumulated_loss / step_count
            avg_pearson = accumulated_pearson / step_count
            total_loss += float(avg_loss)
            total_pearson += float(avg_pearson)
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'pearson': avg_pearson,
    }, optimizer_state, rng_key

def validate(
    model: Any,
    dataloader: MPRADataLoader,
    loss_fn: Any,
    head_name: str = 'mpra_head',
) -> dict:
    """Validate model on validation set.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for validation batches
        loss_fn: Loss function created by model.create_loss_fn_for_head()
        head_name: Name of the custom head
        
    Returns:
        Dictionary with validation metrics
    """
    
    total_loss = 0.0
    total_pearson = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Get predictions (no gradients)
        with model._device_context:
            predictions = model._predict(
                model._params,
                model._state,
                batch['seq'],
                batch['organism_index'],
                negative_strand_mask=jnp.zeros(len(batch['seq']), dtype=bool),
                strand_reindexing=jax.device_put(
                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                    model._device_context._device
                ),
            )
        
        # Get predictions for our head
        head_predictions = predictions[head_name]
        
        # Prepare batch for head's loss method (expects 'targets' key)
        loss_batch = {'targets': batch['y']}
        
        # Use loss function to compute loss and metrics
        loss_dict = loss_fn(head_predictions, loss_batch)
        loss = loss_dict['loss']
        pearson = loss_dict.get('pearson_corr', 0.0)
        
        total_loss += float(loss)
        total_pearson += float(pearson)
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    
    return {
        'val_loss': avg_loss,
        'val_pearson': avg_pearson,
    }

def train(
    model: Any,
    train_loader: MPRADataLoader,
    val_loader: MPRADataLoader | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    head_name: str = 'mpra_head',
    rng_key: jax.Array | None = None,
    gradient_accumulation_steps: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "alphagenome-mpra",
    wandb_name: str | None = None,
    wandb_config: dict | None = None,
    checkpoint_dir: str | None = None,
    save_full_model: bool = False,
    early_stopping_patience: int = 5,
) -> dict:
    """Complete training loop with checkpointing and early stopping.
    
    Args:
        model: CustomAlphaGenomeModel instance (will be modified in-place)
        train_loader: DataLoader for training
        val_loader: Optional DataLoader for validation
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        head_name: Name of the custom head to train
        rng_key: Random key for training
        gradient_accumulation_steps: Number of mini-batches to accumulate gradients
            before updating parameters. Use > 1 to reduce GPU memory usage.
            For example, gradient_accumulation_steps=4 processes batches in 4
            smaller mini-batches, using 1/4 the memory.
        use_wandb: If True, log metrics to Weights & Biases
        wandb_project: Wandb project name
        wandb_name: Wandb run name (defaults to auto-generated)
        wandb_config: Additional config to log to wandb
        checkpoint_dir: Directory to save best model checkpoints. If None, no checkpointing.
            Best model is determined by minimum validation loss.
        save_full_model: If True, saves all parameters including backbone when checkpointing.
            If False (default), only saves custom head parameters (efficient for frozen backbone).
            Only used if checkpoint_dir is provided.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
            Only used if val_loader is provided. Default is 5.
        
    Returns:
        Dictionary with training history
    """
    if optax is None:
        raise ImportError("optax is required for training. Install with: pip install optax")
    
    if use_wandb and not WANDB_AVAILABLE:
        raise ImportError("wandb is required for logging. Install with: pip install wandb")
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # Initialize wandb
    if use_wandb:
        config = {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'head_name': head_name,
            'batch_size': train_loader.batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader) if val_loader else 0,
        }
        if wandb_config:
            config.update(wandb_config)
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config,
        )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(model._params)
    opt_update = optimizer.update
    
    # Create loss function for head
    loss_fn = model.create_loss_fn_for_head(head_name)
    
    # Training history
    history = {
        'train_loss': [],
        'train_pearson': [],
        'val_loss': [],
        'val_pearson': [],
    }
    
    # Early stopping and checkpointing variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
        print(f"Early stopping patience: {early_stopping_patience} epochs")
    if gradient_accumulation_steps > 1:
        print(f"Gradient accumulation steps: {gradient_accumulation_steps} (reduces memory usage)")
    if checkpoint_dir:
        save_type = "full model" if save_full_model else "head only"
        print(f"Checkpoint directory: {checkpoint_dir} (saving {save_type})")
    
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics, optimizer_state, rng_key = train_epoch(
            model, train_loader, optimizer_state, opt_update, rng_key, 
            loss_fn=loss_fn,
            head_name=head_name,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, loss_fn=loss_fn, head_name=head_name)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_pearson'].append(val_metrics['val_pearson'])
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_pearson'].append(train_metrics['pearson'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"train_loss={train_metrics['loss']:.6f}, "
              f"train_pearson={train_metrics['pearson']:.4f}", end="")
        if val_loader:
            print(f", val_loss={val_metrics['val_loss']:.6f}, "
                  f"val_pearson={val_metrics['val_pearson']:.4f}")
        else:
            print()
        
        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_pearson': train_metrics['pearson'],
            }
            if val_loader:
                log_dict['val_loss'] = val_metrics['val_loss']
                log_dict['val_pearson'] = val_metrics['val_pearson']
            wandb.log(log_dict)
        
        # Checkpoint and early stopping logic (only if validation set is provided)
        if val_loader:
            current_val_loss = val_metrics['val_loss']
            
            # Check if this is the best model so far
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save checkpoint if directory is provided
                if checkpoint_dir:
                    save_type = "full model" if save_full_model else "head only"
                    print(f"  → New best model! Saving checkpoint ({save_type}, val_loss: {best_val_loss:.6f})...")
                    model.save_checkpoint(
                        checkpoint_dir,
                        save_full_model=save_full_model
                    )
                else:
                    print(f"  → New best model! (val_loss: {best_val_loss:.6f})")
            else:
                epochs_without_improvement += 1
                
                # Early stopping check
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                    break
    
    if val_loader:
        print(f"\nTraining completed!")
        print(f"Best model: epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    else:
        print("\nTraining completed!")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    return history