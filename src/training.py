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
        head_name: Name of the custom head to train
        gradient_accumulation_steps: Number of mini-batches to accumulate gradients
            before updating parameters. Use > 1 to reduce memory usage.
        
    Returns:
        Tuple of (metrics_dict, updated_optimizer_state, updated_rng_key)
    """
    total_loss = 0.0
    num_batches = 0
    accumulated_grads = None
    accumulated_loss = 0.0
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
            def loss_fn(params):
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
                
                # Compute loss (average over sequence dimension for encoder head)
                # EncoderMPRAHead returns (batch, seq_len//128, num_tracks)
                pred_values = jnp.mean(head_predictions, axis=1)  # (batch, num_tracks)
                
                # Compute MSE loss
                targets = mini_batch['y']
                if targets.ndim == 1:
                    targets = targets[:, None]
                
                loss = jnp.mean((pred_values - targets) ** 2)
                return loss
            
            # Compute gradients for mini-batch
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(model._params)
            
            # Accumulate gradients and loss
            if accumulated_grads is None:
                # Initialize accumulated gradients with first gradient
                accumulated_grads = grads
                accumulated_loss = loss
            else:
                # Add gradients (they'll be averaged when we update)
                accumulated_grads = jax.tree.map(
                    lambda acc, new: acc + new,
                    accumulated_grads,
                    grads
                )
                accumulated_loss += loss
            
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
                
                # Track average loss
                avg_loss = accumulated_loss / gradient_accumulation_steps
                total_loss += float(avg_loss)
                num_batches += 1
                
                # Reset accumulation
                accumulated_grads = None
                accumulated_loss = 0.0
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
            
            # Track average loss
            avg_loss = accumulated_loss / step_count
            total_loss += float(avg_loss)
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
    }, optimizer_state, rng_key

def validate(
    model: Any,
    dataloader: MPRADataLoader,
    head_name: str = 'mpra_head',
) -> dict:
    """Validate model on validation set.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for validation batches
        head_name: Name of the custom head
        
    Returns:
        Dictionary with validation metrics
    """
    total_loss = 0.0
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
        
        # Compute loss
        head_predictions = predictions[head_name]
        pred_values = jnp.mean(head_predictions, axis=1)  # (batch, num_tracks)
        
        targets = batch['y']
        if targets.ndim == 1:
            targets = targets[:, None]
        
        loss = jnp.mean((pred_values - targets) ** 2)
        
        total_loss += float(loss)
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'val_loss': avg_loss,
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
) -> dict:
    """Complete training loop.
    
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
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    if gradient_accumulation_steps > 1:
        print(f"Gradient accumulation steps: {gradient_accumulation_steps} (reduces memory usage)")
    
    for epoch in range(num_epochs):
        # Train epoch
        train_metrics, optimizer_state, rng_key = train_epoch(
            model, train_loader, optimizer_state, opt_update, rng_key, head_name,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, head_name)
            history['val_loss'].append(val_metrics['val_loss'])
        
        history['train_loss'].append(train_metrics['loss'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"train_loss={train_metrics['loss']:.6f}", end="")
        if val_loader:
            print(f", val_loss={val_metrics['val_loss']:.6f}")
        else:
            print()
        
        # Log to wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
            }
            if val_loader:
                log_dict['val_loss'] = val_metrics['val_loss']
            wandb.log(log_dict)
    
    print("\nTraining completed!")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    return history