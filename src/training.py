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


# Cache for head-only forward functions and gradient functions
_cached_functions = {}

def _create_head_only_functions(model: Any, head_name: str, loss_fn: Any = None):
    """Create JIT-compiled head-only forward and gradient functions.
    
    These functions are cached and reused for performance.
    Computes forward + loss in a SINGLE transform for maximum efficiency.
    
    Returns a tuple of (forward_fn, grad_fn).
    """
    cache_key = (id(model), head_name)
    
    if cache_key in _cached_functions:
        return _cached_functions[cache_key]
    
    from alphagenome_ft.embeddings_extended import ExtendedEmbeddings
    import haiku as hk
    from alphagenome_ft import custom_heads as custom_heads_module
    
    head_config = model._head_configs[head_name]
    num_organisms = len(model._metadata)
    
    # Create SINGLE Haiku transform that does BOTH predict AND loss
    # This avoids creating the head twice (once for predict, once for loss)
    @hk.transform_with_state
    def head_forward_and_loss(encoder_output, organism_index, targets=None):
        """Head-only forward pass + loss computation in ONE transform."""
        embeddings = ExtendedEmbeddings(
            embeddings_1bp=None,
            embeddings_128bp=None,
            encoder_output=encoder_output,
        )
        with hk.name_scope('head'):
            head = custom_heads_module.create_custom_head(
                head_name,
                metadata=head_config.metadata,
                num_organisms=num_organisms
            )
            predictions = head.predict(embeddings, organism_index)
            
            # Compute loss if targets provided
            if targets is not None:
                batch = {'targets': targets}
                loss_dict = head.loss(predictions, batch)
                return predictions, loss_dict
            else:
                return predictions, None
    
    # JIT-compiled forward function (no loss)
    @jax.jit
    def forward_fn(params, state, encoder_output, organism_index):
        (predictions, _), new_state = head_forward_and_loss.apply(
            params,
            state,
            None,  # rng
            encoder_output,
            organism_index,
            None  # no targets = no loss
        )
        return predictions
    
    # JIT-compiled gradient function (with loss in same transform!)
    @jax.jit
    def grad_fn(params, state, encoder_output, organism_index, targets):
        """Compute loss and gradients - forward + loss in ONE transform."""
        def loss_fn_inner(p):
            # Single transform call that does BOTH predict and loss!
            (preds, loss_dict), _ = head_forward_and_loss.apply(
                p, state, None, encoder_output, organism_index, targets
            )
            return loss_dict['loss'], loss_dict
        
        (loss_value, loss_dict), grads = jax.value_and_grad(loss_fn_inner, has_aux=True)(params)
        return grads, loss_value, loss_dict
    
    _cached_functions[cache_key] = (forward_fn, grad_fn)
    return forward_fn, grad_fn


def train_epoch(
    model: Any,
    dataloader: MPRADataLoader,
    optimizer_state: Any,
    opt_update: Any,
    rng_key: jax.Array,
    loss_fn: Any,
    head_name: str = 'mpra_head',
    gradient_accumulation_steps: int = 1,
    use_cached_embeddings: bool = False,
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
        use_cached_embeddings: If True, use pre-computed embeddings (faster)
        
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
    
    # Get cached functions if using embeddings (created once, reused)
    if use_cached_embeddings:
        _, grad_fn = _create_head_only_functions(model, head_name, loss_fn)
    
    for batch in dataloader:
        rng_key, subkey = jax.random.split(rng_key)
        
        # Check if using cached embeddings
        if use_cached_embeddings and 'encoder_output' in batch:
            # Use cached embeddings mode with JIT-compiled gradient function
            batch_size = batch['encoder_output'].shape[0]
            mini_batch_size = max(1, batch_size // gradient_accumulation_steps)
            
            for mini_batch_start in range(0, batch_size, mini_batch_size):
                mini_batch_end = min(mini_batch_start + mini_batch_size, batch_size)
                
                # Extract mini-batch data
                encoder_output = batch['encoder_output'][mini_batch_start:mini_batch_end]
                targets = batch['y'][mini_batch_start:mini_batch_end]
                organism_index = batch['organism_index'][mini_batch_start:mini_batch_end]
                
                # Compute gradients using JIT-compiled function (NO function redefinition!)
                with model._device_context:
                    grads, loss_value, loss_dict = grad_fn(
                        model._params,
                        model._state,
                        encoder_output,
                        organism_index,
                        targets
                    )
                
                pearson = loss_dict.get('pearson_corr', 0.0)
                
                # Accumulate gradients and loss
                if accumulated_grads is None:
                    accumulated_grads = grads
                    accumulated_loss = loss_value
                    accumulated_pearson = pearson
                else:
                    accumulated_grads = jax.tree.map(
                        lambda acc, new: acc + new,
                        accumulated_grads,
                        grads
                    )
                    accumulated_loss += loss_value
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
                    
                    # Track metrics
                    avg_loss = accumulated_loss / gradient_accumulation_steps
                    avg_pearson = accumulated_pearson / gradient_accumulation_steps
                    total_loss += float(avg_loss)
                    total_pearson += float(avg_pearson)
                    num_batches += 1
                    
                    # Reset accumulators
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accumulated_pearson = 0.0
                    step_count = 0
        else:
            # Normal mode: use sequences
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
    use_cached_embeddings: bool = False,
) -> dict:
    """Validate model on validation set. Note these are batch-averaged metrics.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for validation batches
        loss_fn: Loss function created by model.create_loss_fn_for_head()
        head_name: Name of the custom head
        use_cached_embeddings: If True, use cached encoder embeddings from batch
        
    Returns:
        Dictionary with validation metrics
    """
    
    total_loss = 0.0
    total_pearson = 0.0
    num_batches = 0
    additional_metrics = {}  # Track additional metrics like dev_pearson, hk_pearson
    
    # Get cached forward function if using embeddings
    if use_cached_embeddings:
        forward_fn, _ = _create_head_only_functions(model, head_name, loss_fn)
    
    for batch in dataloader:
        # Get predictions (no gradients)
        with model._device_context:
            if use_cached_embeddings and 'encoder_output' in batch:
                # Use JIT-compiled forward function
                predictions = forward_fn(
                    model._params,
                    model._state,
                    batch['encoder_output'],
                    batch['organism_index'],
                )
                # Wrap in dict for consistency
                predictions = {head_name: predictions}
            else:
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
        
        # Accumulate additional metrics (e.g., dev_pearson, hk_pearson for DeepSTARR)
        for key, value in loss_dict.items():
            if key not in ['loss', 'mse', 'pearson_corr']:
                if key not in additional_metrics:
                    additional_metrics[key] = 0.0
                additional_metrics[key] += float(value)
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    
    result = {
        'val_loss': avg_loss,
        'val_pearson': avg_pearson,
    }
    
    # Add averaged additional metrics
    for key, value in additional_metrics.items():
        result[f'val_{key}'] = value / num_batches if num_batches > 0 else 0.0
    
    return result

def test(
    model: Any,
    dataloader: MPRADataLoader,
    loss_fn: Any,
    head_name: str = 'mpra_head',
    use_cached_embeddings: bool = False,
) -> dict:
    """Test model on test set. Note these are batch-averaged metrics.
    
    Args:
        model: CustomAlphaGenomeModel instance
        dataloader: DataLoader for test batches
        loss_fn: Loss function created by model.create_loss_fn_for_head()
        head_name: Name of the custom head
        use_cached_embeddings: If True, use cached encoder embeddings from batch
        
    Returns:
        Dictionary with test metrics
    """
    
    total_loss = 0.0
    total_pearson = 0.0
    num_batches = 0
    additional_metrics = {}  # Track additional metrics like dev_pearson, hk_pearson
    
    # Get cached forward function if using embeddings
    if use_cached_embeddings:
        forward_fn, _ = _create_head_only_functions(model, head_name, loss_fn)
    
    for batch in dataloader:
        # Get predictions (no gradients)
        with model._device_context:
            if use_cached_embeddings and 'encoder_output' in batch:
                # Use JIT-compiled forward function
                predictions = forward_fn(
                    model._params,
                    model._state,
                    batch['encoder_output'],
                    batch['organism_index'],
                )
                # Wrap in dict for consistency
                predictions = {head_name: predictions}
            else:
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
        
        # Accumulate additional metrics (e.g., dev_pearson, hk_pearson for DeepSTARR)
        for key, value in loss_dict.items():
            if key not in ['loss', 'mse', 'pearson_corr']:
                if key not in additional_metrics:
                    additional_metrics[key] = 0.0
                additional_metrics[key] += float(value)
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    
    result = {
        'test_loss': avg_loss,
        'test_pearson': avg_pearson,
    }
    
    # Add averaged additional metrics
    for key, value in additional_metrics.items():
        result[f'test_{key}'] = value / num_batches if num_batches > 0 else 0.0
    
    return result    

def _train_stage(
    model: Any,
    train_loader: MPRADataLoader,
    val_loader: MPRADataLoader | None,
    test_loader: MPRADataLoader | None,
    num_epochs: int,
    learning_rate: float,
    head_name: str,
    optimizer_state: Any,
    opt_update: Any,
    rng_key: jax.Array,
    loss_fn: Any,
    gradient_accumulation_steps: int,
    use_wandb: bool,
    checkpoint_dir: str | None,
    save_full_model: bool,
    early_stopping_patience: int,
    val_eval_frequency: int,
    test_eval_frequency: int,
    stage_name: str = "Stage 1",
    start_epoch: int = 0,
    use_cached_embeddings: bool = False,
) -> tuple[dict, Any, jax.Array, int, float]:
    """Train a single stage of training.
    
    Returns:
        Tuple of (history, optimizer_state, rng_key, best_epoch, best_val_loss)
    """
    # Training history for this stage
    history = {
        'train_loss': [],
        'train_pearson': [],
        'val_loss': [],
        'val_pearson': [],
        'test_loss': [],
        'test_pearson': [],
    }
    
    # Early stopping and checkpointing variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = start_epoch
    
    # Step counter for WandB logging (tracks training steps, not epochs)
    global_step = start_epoch * len(train_loader) if start_epoch > 0 else 0
    
    print(f"\n{'=' * 80}")
    print(f"{stage_name} Training")
    print(f"{'=' * 80}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
        print(f"Validation eval frequency: {val_eval_frequency} per epoch")
        print(f"Early stopping patience: {early_stopping_patience} epochs")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
        print(f"Test eval frequency: {test_eval_frequency} per epoch")
    if gradient_accumulation_steps > 1:
        print(f"Gradient accumulation steps: {gradient_accumulation_steps} (reduces memory usage)")
    if checkpoint_dir:
        save_type = "full model" if save_full_model else "head only"
        print(f"Checkpoint directory: {checkpoint_dir} (saving {save_type})")
    print(f"{'=' * 80}\n")
    
    # Get cached gradient function if using embeddings (created once, reused)
    cached_grad_fn = None
    if use_cached_embeddings:
        _, cached_grad_fn = _create_head_only_functions(model, head_name, loss_fn)
    
    for epoch in range(num_epochs):
        # Track metrics for this epoch
        epoch_val_losses = []
        epoch_val_pearsons = []
        epoch_test_losses = []
        epoch_test_pearsons = []
        
        # Calculate evaluation intervals for this epoch
        num_train_batches = len(train_loader)
        
        # Calculate validation evaluation points
        if val_loader and val_eval_frequency > 1:
            val_eval_interval = max(1, num_train_batches // val_eval_frequency)
            val_eval_points = [i * val_eval_interval for i in range(1, val_eval_frequency + 1)]
            val_eval_points = [min(p, num_train_batches) for p in val_eval_points]
            val_eval_points = sorted(set(val_eval_points))
        else:
            val_eval_points = [num_train_batches]  # Only at end of epoch
        
        # Calculate test evaluation points
        if test_loader and test_eval_frequency > 1:
            test_eval_interval = max(1, num_train_batches // test_eval_frequency)
            test_eval_points = [i * test_eval_interval for i in range(1, test_eval_frequency + 1)]
            test_eval_points = [min(p, num_train_batches) for p in test_eval_points]
            test_eval_points = sorted(set(test_eval_points))
        else:
            test_eval_points = [num_train_batches]  # Only at end of epoch
        
        # Train epoch with periodic evaluation
        batch_idx = 0
        train_losses = []
        train_pearsons = []
        
        for batch in train_loader:
            rng_key, subkey = jax.random.split(rng_key)
            
            # Process batch with gradient accumulation
            if use_cached_embeddings and 'encoder_output' in batch:
                batch_size = batch['encoder_output'].shape[0]
            else:
                batch_size = batch['seq'].shape[0]
            mini_batch_size = max(1, batch_size // gradient_accumulation_steps)
            
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_pearson = 0.0
            step_count = 0
            
            for mini_batch_start in range(0, batch_size, mini_batch_size):
                mini_batch_end = min(mini_batch_start + mini_batch_size, batch_size)
                
                if use_cached_embeddings and 'encoder_output' in batch:
                    # Extract mini-batch data
                    encoder_output = batch['encoder_output'][mini_batch_start:mini_batch_end]
                    targets = batch['y'][mini_batch_start:mini_batch_end]
                    organism_index = batch['organism_index'][mini_batch_start:mini_batch_end]
                    
                    # Use cached JIT-compiled gradient function
                    with model._device_context:
                        grads, loss, loss_dict = cached_grad_fn(
                            model._params,
                            model._state,
                            encoder_output,
                            organism_index,
                            targets
                        )
                    pearson = loss_dict.get('pearson_corr', 0.0)
                else:
                    mini_batch = {
                        'seq': batch['seq'][mini_batch_start:mini_batch_end],
                        'y': batch['y'][mini_batch_start:mini_batch_end],
                        'organism_index': batch['organism_index'][mini_batch_start:mini_batch_end],
                    }
                    
                    def loss_fn_inner(params):
                        with model._device_context:
                            predictions = model._predict(
                                params, model._state, mini_batch['seq'], mini_batch['organism_index'],
                                negative_strand_mask=jnp.zeros(len(mini_batch['seq']), dtype=bool),
                                strand_reindexing=jax.device_put(
                                    model._metadata[dna_model.Organism.HOMO_SAPIENS].strand_reindexing,
                                    model._device_context._device
                                ),
                            )
                        head_predictions = predictions[head_name]
                        loss_batch = {'targets': mini_batch['y']}
                        loss_dict = loss_fn(head_predictions, loss_batch)
                        return loss_dict['loss'], loss_dict
                    
                    grad_fn = jax.value_and_grad(loss_fn_inner, has_aux=True)
                    (loss, loss_dict), grads = grad_fn(model._params)
                    pearson = loss_dict.get('pearson_corr', 0.0)
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                    accumulated_loss = loss
                    accumulated_pearson = pearson
                else:
                    accumulated_grads = jax.tree.map(lambda acc, new: acc + new, accumulated_grads, grads)
                    accumulated_loss += loss
                    accumulated_pearson += pearson
                
                step_count += 1
                
                if step_count >= gradient_accumulation_steps:
                    accumulated_grads = jax.tree.map(lambda g: g / gradient_accumulation_steps, accumulated_grads)
                    updates, optimizer_state = opt_update(accumulated_grads, optimizer_state, model._params)
                    model._params = optax.apply_updates(model._params, updates)
                    avg_loss = accumulated_loss / gradient_accumulation_steps
                    avg_pearson = accumulated_pearson / gradient_accumulation_steps
                    train_losses.append(float(avg_loss))
                    train_pearsons.append(float(avg_pearson))
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accumulated_pearson = 0.0
                    step_count = 0
            
            # Handle remaining gradients
            if step_count > 0:
                accumulated_grads = jax.tree.map(lambda g: g / step_count, accumulated_grads)
                updates, optimizer_state = opt_update(accumulated_grads, optimizer_state, model._params)
                model._params = optax.apply_updates(model._params, updates)
                avg_loss = accumulated_loss / step_count
                avg_pearson = accumulated_pearson / step_count
                train_losses.append(float(avg_loss))
                train_pearsons.append(float(avg_pearson))
            
            batch_idx += 1
            global_step += 1
            
            # Evaluate validation at specified intervals
            if val_loader and batch_idx in val_eval_points:
                val_metrics = validate(model, val_loader, loss_fn=loss_fn, head_name=head_name, use_cached_embeddings=use_cached_embeddings)
                epoch_val_losses.append(val_metrics['val_loss'])
                epoch_val_pearsons.append(val_metrics['val_pearson'])
                
                # Log to WandB immediately at evaluation point
                if use_wandb:
                    log_dict = {
                        'step': global_step,
                        'epoch': start_epoch + epoch + (batch_idx / len(train_loader)),  # Fractional epoch
                        'stage': stage_name,
                        'val_loss': val_metrics['val_loss'],
                        'val_pearson': val_metrics['val_pearson'],
                    }
                    # Include any additional metrics (e.g., val_dev_pearson, val_hk_pearson)
                    for key, value in val_metrics.items():
                        if key not in ['val_loss', 'val_pearson']:
                            log_dict[key] = value
                    # Include running average of train metrics if available
                    if train_losses:
                        log_dict['train_loss'] = sum(train_losses) / len(train_losses)
                    if train_pearsons:
                        log_dict['train_pearson'] = sum(train_pearsons) / len(train_pearsons)
                    wandb.log(log_dict)
            
            # Evaluate test at specified intervals
            if test_loader and batch_idx in test_eval_points:
                test_metrics = test(model, test_loader, loss_fn=loss_fn, head_name=head_name, use_cached_embeddings=use_cached_embeddings)
                epoch_test_losses.append(test_metrics['test_loss'])
                epoch_test_pearsons.append(test_metrics['test_pearson'])
                
                # Log to WandB immediately at evaluation point
                if use_wandb:
                    log_dict = {
                        'step': global_step,
                        'epoch': start_epoch + epoch + (batch_idx / len(train_loader)),  # Fractional epoch
                        'stage': stage_name,
                        'test_loss': test_metrics['test_loss'],
                        'test_pearson': test_metrics['test_pearson'],
                    }
                    # Include any additional metrics (e.g., test_dev_pearson, test_hk_pearson)
                    for key, value in test_metrics.items():
                        if key not in ['test_loss', 'test_pearson']:
                            log_dict[key] = value
                    # Include running average of train metrics if available
                    if train_losses:
                        log_dict['train_loss'] = sum(train_losses) / len(train_losses)
                    if train_pearsons:
                        log_dict['train_pearson'] = sum(train_pearsons) / len(train_pearsons)
                    wandb.log(log_dict)
        
        # Average metrics for this epoch
        train_metrics = {
            'loss': sum(train_losses) / len(train_losses) if train_losses else 0.0,
            'pearson': sum(train_pearsons) / len(train_pearsons) if train_pearsons else 0.0,
        }
        
        # Use last evaluation metrics
        val_metrics = None
        if epoch_val_losses:
            val_metrics = {
                'val_loss': epoch_val_losses[-1],
                'val_pearson': epoch_val_pearsons[-1],
            }
        
        test_metrics = None
        if epoch_test_losses:
            test_metrics = {
                'test_loss': epoch_test_losses[-1],
                'test_pearson': epoch_test_pearsons[-1],
            }
        
        # Store in history
        history['train_loss'].append(train_metrics['loss'])
        history['train_pearson'].append(train_metrics['pearson'])
        if val_metrics:
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_pearson'].append(val_metrics['val_pearson'])
        if test_metrics:
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_pearson'].append(test_metrics['test_pearson'])
        
        # Print progress
        current_epoch = start_epoch + epoch + 1
        print(f"Epoch {current_epoch}: "
              f"train_loss={train_metrics['loss']:.6f}, "
              f"train_pearson={train_metrics['pearson']:.4f}", end="")
        if val_metrics:
            print(f", val_loss={val_metrics['val_loss']:.6f}, "
                  f"val_pearson={val_metrics['val_pearson']:.4f}", end="")
        if test_metrics:
            print(f", test_loss={test_metrics['test_loss']:.6f}, "
                  f"test_pearson={test_metrics['test_pearson']:.4f}", end="")
        print()
        
        # Log to wandb at end of epoch (only if we haven't already logged at evaluation points)
        # This ensures we log at least once per epoch even if no intermediate evaluations occurred
        if use_wandb:
            log_dict = {
                'step': global_step,
                'epoch': current_epoch,
                'stage': stage_name,
                'train_loss': train_metrics['loss'],
                'train_pearson': train_metrics['pearson'],
            }
            if val_metrics:
                log_dict['val_loss'] = val_metrics['val_loss']
                log_dict['val_pearson'] = val_metrics['val_pearson']
                # Include additional val metrics
                for key, value in val_metrics.items():
                    if key not in ['val_loss', 'val_pearson']:
                        log_dict[key] = value
            if test_metrics:
                log_dict['test_loss'] = test_metrics['test_loss']
                log_dict['test_pearson'] = test_metrics['test_pearson']
                 # Include additional test metrics
                for key, value in test_metrics.items():
                    if key not in ['test_loss', 'test_pearson']:
                        log_dict[key] = value
            wandb.log(log_dict)
        
        # Checkpoint and early stopping logic (only if validation set is provided)
        if val_metrics:
            current_val_loss = val_metrics['val_loss']
            
            # Check if this is the best model so far
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = current_epoch
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
                    print(f"\nEarly stopping triggered after {current_epoch} epochs!")
                    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                    break
    
    return history, optimizer_state, rng_key, best_epoch, best_val_loss


def train(
    model: Any,
    train_loader: MPRADataLoader,
    val_loader: MPRADataLoader | None = None,
    test_loader: MPRADataLoader | None = None,
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
    val_eval_frequency: int = 1,
    test_eval_frequency: int = 1,
    second_stage_lr: float | None = None,
    resume_from_stage2: bool = False,
    gradient_clip: float | None = None,
    use_cached_embeddings: bool = False,
    lr_scheduler: str | None = None,
    save_test_results: str | None = None,
) -> dict:
    """Complete training loop with checkpointing and early stopping.
    
    Supports two-stage training:
    - Stage 1: Train with frozen backbone (head only)
    - Stage 2: Unfreeze encoder and continue training from best Stage 1 checkpoint
    
    Args:
        model: CustomAlphaGenomeModel instance (will be modified in-place)
        train_loader: DataLoader for training
        val_loader: Optional DataLoader for validation
        test_loader: Optional DataLoader for test
        num_epochs: Number of training epochs for Stage 1. Stage 2 uses half this many.
        learning_rate: Learning rate for Stage 1 optimizer
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
            Note: When two-stage training is enabled (second_stage_lr provided), Stage 1 will
            automatically save the full model regardless of this flag, as Stage 2 requires the
            full model to unfreeze the encoder.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
            Only used if val_loader is provided. Default is 5. Applies to both stages.
        val_eval_frequency: Number of times to evaluate on validation set per epoch.
            Default is 1 (evaluate once at end of epoch). Set to N to evaluate N times.
        test_eval_frequency: Number of times to evaluate on test set per epoch.
            Default is 1 (evaluate once at end of epoch). Set to N to evaluate N times.
        second_stage_lr: If provided, enables two-stage training. After Stage 1 completes,
            loads best checkpoint, unfreezes encoder, and continues training with this
            learning rate for num_epochs//2 epochs.
        resume_from_stage2: If True, skip Stage 1 training and resume directly from Stage 2.
            Requires checkpoint_dir to contain a Stage 1 checkpoint. second_stage_lr must be provided.
            Default is False.
        gradient_clip: If provided, clips gradients by global norm to this value.
            Helps stabilize training and prevent gradient explosion. Try 1.0 or 5.0. Default is None.
        
    Returns:
        Dictionary with training history (combined from both stages if second_stage_lr is provided)
    """
    if optax is None:
        raise ImportError("optax is required for training. Install with: pip install optax")
    
    if use_wandb and not WANDB_AVAILABLE:
        raise ImportError("wandb is required for logging. Install with: pip install wandb")
    
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)
    
    # Validate resume_from_stage2
    if resume_from_stage2:
        if second_stage_lr is None:
            raise ValueError("resume_from_stage2 requires second_stage_lr to be provided")
        if checkpoint_dir is None:
            raise ValueError("resume_from_stage2 requires checkpoint_dir to be provided")
    
    # Determine if two-stage training
    enable_second_stage = second_stage_lr is not None
    stage1_epochs = (num_epochs // 4) * 1 if enable_second_stage else num_epochs
    stage2_epochs = (num_epochs // 4) * 3 if enable_second_stage else 0
    
    # Initialize wandb
    if use_wandb:
        config = {
            'num_epochs': num_epochs,
            'stage1_epochs': stage1_epochs,
            'learning_rate': learning_rate,
            'head_name': head_name,
            'batch_size': train_loader.batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader) if val_loader else 0,
            'test_batches': len(test_loader) if test_loader else 0,
            'val_eval_frequency': val_eval_frequency,
            'test_eval_frequency': test_eval_frequency,
            'two_stage_training': enable_second_stage,
        }
        if enable_second_stage:
            config['stage2_epochs'] = stage2_epochs
            config['second_stage_lr'] = second_stage_lr
        if wandb_config:
            config.update(wandb_config)
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config,
        )
    
    # Get optimizer and regularization settings from config
    # These can come from wandb_config or be passed directly
    optimizer_type = 'adam'  # default
    weight_decay = None
    if wandb_config:
        optimizer_type = wandb_config.get('optimizer', 'adam')
        weight_decay = wandb_config.get('weight_decay', None)
    
    # Create learning rate schedule if specified
    def create_lr_schedule(base_lr: float, total_steps: int):
        """Create learning rate schedule based on lr_scheduler parameter."""
        if lr_scheduler is None:
            return base_lr
        elif lr_scheduler == 'cosine':
            # Cosine decay with warmup (10% of total steps)
            warmup_steps = int(0.1 * total_steps)
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=base_lr * 0.01
            )
        elif lr_scheduler == 'plateau':
            # Reduce LR by 0.5 at 50% and 75% of training
            boundaries = [int(0.5 * total_steps), int(0.75 * total_steps)]
            values = [base_lr, base_lr * 0.5, base_lr * 0.25]
            return optax.piecewise_constant_schedule(
                init_value=values[0],
                boundaries_and_scales={b: v/values[i] for i, (b, v) in enumerate(zip(boundaries, values[1:]))}
            )
        else:
            print(f"Warning: Unknown lr_scheduler '{lr_scheduler}', using constant LR")
            return base_lr
    
    # Create optimizer with gradient clipping
    # If resuming from Stage 2, we'll create Stage 2 optimizer later
    def create_optimizer(lr_or_schedule):
        """Create optimizer with optional gradient clipping and weight decay.
        
        Weight decay is applied differently for Adam vs AdamW:
        - AdamW: built-in decoupled weight decay
        - Adam: L2 regularization via optax.add_decayed_weights
        
        Args:
            lr_or_schedule: Learning rate (float) or schedule (callable)
        """
        if optimizer_type.lower() == 'adamw':
            # AdamW has built-in decoupled weight decay
            if weight_decay is not None:
                optimizer = optax.adamw(learning_rate=lr_or_schedule, weight_decay=weight_decay)
            else:
                optimizer = optax.adamw(learning_rate=lr_or_schedule)
        else:  # adam
            # Base Adam optimizer
            optimizer = optax.adam(lr_or_schedule)
            
            # Add L2 regularization if weight_decay specified
            if weight_decay is not None:
                optimizer = optax.chain(
                    optax.add_decayed_weights(weight_decay),
                    optimizer
                )
        
        # Add gradient clipping if specified
        if gradient_clip is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(gradient_clip),
                optimizer
            )
        
        return optimizer
    
    # Calculate total steps for LR scheduler
    steps_per_epoch = len(train_loader)
    total_stage1_steps = stage1_epochs * steps_per_epoch
    
    if not resume_from_stage2:
        # Create LR schedule and optimizer for Stage 1
        lr_schedule = create_lr_schedule(learning_rate, total_stage1_steps)
        optimizer = create_optimizer(lr_schedule)
        optimizer_state = optimizer.init(model._params)
        opt_update = optimizer.update
    else:
        # Will be initialized in Stage 2 section
        optimizer_state = None
        opt_update = None
    
    # Create loss function for head
    loss_fn = model.create_loss_fn_for_head(head_name)
    
    # Combined training history
    history = {
        'train_loss': [],
        'train_pearson': [],
        'val_loss': [],
        'val_pearson': [],
        'test_loss': [],
        'test_pearson': [],
    }
    
    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    if resume_from_stage2:
        print(f"Resuming from Stage 2 (skipping Stage 1)")
        print(f"Stage 2: {stage2_epochs} epochs (unfrozen encoder, LR={second_stage_lr})")
    elif enable_second_stage:
        print(f"Two-stage training enabled")
        print(f"Stage 1: {stage1_epochs} epochs (frozen backbone, LR={learning_rate})")
        print(f"Stage 2: {stage2_epochs} epochs (unfrozen encoder, LR={second_stage_lr})")
    else:
        print(f"Single-stage training: {stage1_epochs} epochs (LR={learning_rate})")
    print(f"{'=' * 80}\n")
    
    # Create separate checkpoint directory for Stage 1
    from pathlib import Path
    stage1_checkpoint_dir = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir).resolve()  # Convert to absolute path
        stage1_checkpoint_dir = str(checkpoint_path / 'stage1')
    
    # Determine save_full_model for Stage 1
    # If two-stage training is enabled, Stage 1 must save full model for Stage 2 to load
    stage1_save_full_model = save_full_model or enable_second_stage
    if enable_second_stage and not save_full_model:
        print(f"Note: Two-stage training enabled - Stage 1 will save full model (required for Stage 2)")
    
    # ========================================================================
    # STAGE 1: Train with frozen backbone (skip if resuming from Stage 2)
    # ========================================================================
    if not resume_from_stage2:
        stage1_history, optimizer_state, rng_key, stage1_best_epoch, stage1_best_val_loss = _train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=stage1_epochs,
            learning_rate=learning_rate,
            head_name=head_name,
            optimizer_state=optimizer_state,
            opt_update=opt_update,
            rng_key=rng_key,
            loss_fn=loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb,
            checkpoint_dir=stage1_checkpoint_dir,
            save_full_model=stage1_save_full_model,
            early_stopping_patience=early_stopping_patience,
            val_eval_frequency=val_eval_frequency,
            test_eval_frequency=test_eval_frequency,
            stage_name="Stage 1 (Frozen Backbone)",
            start_epoch=0,
            use_cached_embeddings=use_cached_embeddings,
        )
        
        # Merge Stage 1 history
        for key in history:
            history[key].extend(stage1_history[key])
    else:
        # When resuming from Stage 2, we need to load Stage 1 checkpoint
        # Set dummy values for stage1_best_epoch and stage1_best_val_loss
        stage1_best_epoch = 0
        stage1_best_val_loss = float('inf')
        print(f"Skipping Stage 1 training (resuming from Stage 2)")
    
    # ========================================================================
    # STAGE 2: Unfreeze encoder and continue training
    # ========================================================================
    if enable_second_stage:
        if resume_from_stage2:
            print(f"\n{'=' * 80}")
            print("RESUMING FROM STAGE 2")
            print(f"{'=' * 80}")
            print(f"Loading Stage 1 checkpoint and proceeding to Stage 2...")
            print(f"{'=' * 80}\n")
        else:
            print(f"\n{'=' * 80}")
            print("STAGE 1 COMPLETE")
            print(f"{'=' * 80}")
            print(f"Best validation loss: {stage1_best_val_loss:.6f} at epoch {stage1_best_epoch}")
            print(f"{'=' * 80}\n")
        
        # Load best checkpoint from Stage 1
        if stage1_checkpoint_dir:
            print(f"Loading best checkpoint from Stage 1...")
            import json
            import orbax.checkpoint as ocp
            
            stage1_checkpoint_path = Path(stage1_checkpoint_dir)
            checkpoint_path = stage1_checkpoint_path / 'checkpoint'
            
            if not checkpoint_path.exists():
                raise ValueError(f"Stage 1 checkpoint not found at {checkpoint_path}")
            
            # Load checkpoint config to determine if it's full model or head-only
            config_path = stage1_checkpoint_path / 'config.json'
            if not config_path.exists():
                raise ValueError(f"Config file not found at {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            save_full_model = config.get('save_full_model', False)
            
            # Load checkpoint parameters using Orbax
            checkpointer = ocp.StandardCheckpointer()
            loaded_params, loaded_state = checkpointer.restore(checkpoint_path)
            
            if save_full_model:
                # Full model checkpoint - replace all parameters
                model._params = loaded_params
                model._state = loaded_state
            else:
                # Head-only checkpoint - merge head parameters into existing model
                # Handle all possible parameter structures
                def merge_head_params(model_params, loaded_head_params):
                    """Merge loaded head parameters into model parameters."""
                    import copy
                    merged = copy.deepcopy(model_params)
                    
                    # Structure 1: head/{head_name}/... (use_encoder_output=True mode)
                    # This is a flat dict with keys like 'head/mpra_head/~predict/hidden'
                    if isinstance(loaded_head_params, dict):
                        # Check if we have flat keys starting with 'head/'
                        head_keys = {k: v for k, v in loaded_head_params.items() 
                                    if isinstance(k, str) and k.startswith('head/')}
                        if head_keys:
                            # Merge flat keys directly
                            for key, value in head_keys.items():
                                merged[key] = value
                    
                    # Structure 2: alphagenome/head (encoder-only mode, nested)
                    if 'alphagenome/head' in loaded_head_params:
                        if 'alphagenome/head' not in merged:
                            merged['alphagenome/head'] = {}
                        
                        for head_name, head_params in loaded_head_params['alphagenome/head'].items():
                            merged['alphagenome/head'][head_name] = head_params
                    
                    # Structure 3: alphagenome -> head (standard mode, nested)
                    if 'alphagenome' in loaded_head_params:
                        if isinstance(loaded_head_params['alphagenome'], dict):
                            if 'head' in loaded_head_params['alphagenome']:
                                if 'alphagenome' not in merged:
                                    merged['alphagenome'] = {}
                                if not isinstance(merged['alphagenome'], dict):
                                    merged['alphagenome'] = {}
                                if 'head' not in merged['alphagenome']:
                                    merged['alphagenome']['head'] = {}
                                
                                for head_name, head_params in loaded_head_params['alphagenome']['head'].items():
                                    merged['alphagenome']['head'][head_name] = head_params
                    
                    return merged
                
                model._params = merge_head_params(model._params, loaded_params)
                model._state = merge_head_params(model._state, loaded_state)
            
            # Re-put on device
            device = model._device_context._device
            model._params = jax.device_put(model._params, device)
            model._state = jax.device_put(model._state, device)
            
            print("✓ Best checkpoint loaded from Stage 1")
            
            # Validate checkpoint by evaluating on validation set
            if val_loader:
                print("\nValidating loaded checkpoint...")
                val_metrics = validate(model, val_loader, loss_fn=loss_fn, head_name=head_name)
                print(f"  Validation loss: {val_metrics['val_loss']:.6f}")
                print(f"  Validation Pearson: {val_metrics['val_pearson']:.4f}")
                print(f"  ✓ Checkpoint validation complete")
                print()
        
        # Unfreeze encoder
        print("\nUnfreezing encoder weights...")
        # Unfreeze sequence encoder (path is 'alphagenome/sequence_encoder/...')
        model.unfreeze_parameters(unfreeze_prefixes=['sequence_encoder'])
        print("✓ Encoder unfrozen")
        
        # Create new optimizer for Stage 2 with different learning rate (same regularization)
        print(f"\nCreating optimizer for Stage 2 (LR={second_stage_lr})...")
        total_stage2_steps = stage2_epochs * steps_per_epoch
        lr_schedule_stage2 = create_lr_schedule(second_stage_lr, total_stage2_steps)
        optimizer_stage2 = create_optimizer(lr_schedule_stage2)
        optimizer_state = optimizer_stage2.init(model._params)
        opt_update = optimizer_stage2.update
        print("✓ Optimizer created")
        
        # Create separate checkpoint directory for Stage 2
        stage2_checkpoint_dir = None
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir).resolve()  # Convert to absolute path
            stage2_checkpoint_dir = str(checkpoint_path / 'stage2')
        
        # Train Stage 2
        # Always save full model in Stage 2 since encoder is trainable
        stage2_history, optimizer_state, rng_key, stage2_best_epoch, stage2_best_val_loss = _train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=stage2_epochs,
            learning_rate=second_stage_lr,
            head_name=head_name,
            optimizer_state=optimizer_state,
            opt_update=opt_update,
            rng_key=rng_key,
            loss_fn=loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb,
            checkpoint_dir=stage2_checkpoint_dir,
            save_full_model=True,  # Always save full model in Stage 2 since encoder is trainable
            early_stopping_patience=early_stopping_patience,
            val_eval_frequency=val_eval_frequency,
            test_eval_frequency=test_eval_frequency,
            stage_name="Stage 2 (Unfrozen Encoder)",
            start_epoch=stage1_best_epoch,
            use_cached_embeddings=False,  # Stage 2 trains encoder, so can't use cached embeddings
        )
        
        # Merge Stage 2 history
        for key in history:
            history[key].extend(stage2_history[key])
        
        print(f"\n{'=' * 80}")
        print("STAGE 2 COMPLETE")
        print(f"{'=' * 80}")
        print(f"Best validation loss: {stage2_best_val_loss:.6f} at epoch {stage2_best_epoch}")
        print(f"{'=' * 80}\n")
    
    # Final summary
    if val_loader:
        overall_best_val_loss = min(history['val_loss']) if history['val_loss'] else None
        overall_best_epoch = history['val_loss'].index(overall_best_val_loss) + 1 if overall_best_val_loss else None
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 80}")
        if overall_best_val_loss is not None:
            print(f"Overall best validation loss: {overall_best_val_loss:.6f} at epoch {overall_best_epoch}")
        print(f"{'=' * 80}")
    else:
        print("\nTraining completed!")
    
    # Save test results to CSV if requested
    if save_test_results and test_loader and history['test_loss']:
        from pathlib import Path
        import csv
        
        results_file = Path(save_test_results)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get best test metrics
        best_test_loss = min(history['test_loss'])
        best_test_epoch = history['test_loss'].index(best_test_loss) + 1
        best_test_pearson = history['test_pearson'][best_test_epoch - 1] if best_test_epoch <= len(history['test_pearson']) else history['test_pearson'][-1]
        final_test_loss = history['test_loss'][-1]
        final_test_pearson = history['test_pearson'][-1]
        
        # Write results
        file_exists = results_file.exists()
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run_name', 'cell_type', 'best_test_loss', 'best_test_pearson', 
                'best_test_epoch', 'final_test_loss', 'final_test_pearson'
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'run_name': wandb_name if wandb_name else 'unnamed',
                'cell_type': wandb_config.get('cell_type', 'unknown') if wandb_config else 'unknown',
                'best_test_loss': f'{best_test_loss:.6f}',
                'best_test_pearson': f'{best_test_pearson:.4f}',
                'best_test_epoch': best_test_epoch,
                'final_test_loss': f'{final_test_loss:.6f}',
                'final_test_pearson': f'{final_test_pearson:.4f}',
            })
        print(f"\n✓ Test results saved to {results_file}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    return history