"""
MPRA Heads with flexible access to different architecture levels.

Three head types available:

1. MPRAHead - Standard head with configurable embedding mode
   - embedding_mode='1bp': Uses decoder output (encoder + transformer via decoder)
   - embedding_mode='128bp': Uses transformer output directly (global context)

2. EncoderMPRAHead - Uses raw encoder output (BEFORE transformer)
   - Requires ExtendedEmbeddings with encoder_output
   - No fallback - fails if encoder_output not available
   - Pure CNN features, no global attention

Example usage:

# Standard head with 1bp embeddings (default - has both encoder + transformer)
register_custom_head(
    'mpra_1bp',
    MPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_1bp',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={
            'center_window_size': 128,
            'pooling_type': 'sum',
            'embedding_mode': '1bp',  # Decoder output (local + global)
        }
    )
)

# Standard head with 128bp embeddings (transformer output - global context)
register_custom_head(
    'mpra_128bp',
    MPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_128bp',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={
            'center_window_size': 128,
            'pooling_type': 'sum',
            'embedding_mode': '128bp',  # Transformer output
        }
    )
)

# Encoder-only head (requires ExtendedEmbeddings)
register_custom_head(
    'mpra_encoder',
    EncoderMPRAHead,
    HeadConfig(
        type=HeadType.GENOME_TRACKS,
        name='mpra_encoder',
        output_type=dna_output.OutputType.RNA_SEQ,
        num_tracks=1,
        metadata={
            'center_window_size': 128,
            'pooling_type': 'sum',
        }
    )
)
"""

import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead
from alphagenome_research.model import layers


class MPRAHead(CustomHead):
    """Custom head for predicting MPRA activity with flexible architecture access.
    
    Can use:
    - 1bp embeddings (encoder + transformer via decoder)
    - 128bp embeddings (pure transformer output)
    
    Configuration:
    - center_bp: Center region to average over, in BASE PAIRS (default: 128bp)
                 Converted to appropriate resolution based on embedding_mode.
                 For 1bp mode: uses center_bp directly
                 For 128bp mode: center_bp/128 positions
    - pooling_type: How to pool over center region ('mean', 'sum', or 'max')
    - embedding_mode: Which embeddings to use ('1bp' or '128bp')
    """
    
    def __init__(
        self, 
        *, 
        name, 
        output_type, 
        num_tracks, 
        num_organisms, 
        metadata,
    ):
        super().__init__(
            name=name,
            num_tracks=num_tracks,
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )
        
        # Get center region in base pairs (default: 256bp)
        # Note: 'center_bp' is preferred, but 'center_window_size' is kept for backward compatibility
        self._center_bp = (
            metadata.get('center_bp', metadata.get('center_window_size', 256)) 
            if metadata else 128
        )
        
        # Get pooling_type from metadata (default: 'sum')
        pooling_type = (
            metadata.get('pooling_type', 'sum') if metadata else 'sum'
        )
        assert pooling_type in ['sum', 'mean', 'max'], f'Invalid pooling type: {pooling_type}'
        self._pooling_type = pooling_type
        
        # Choose which embeddings to use: '1bp' or '128bp'
        self._embedding_mode = metadata.get('embedding_mode', '1bp') if metadata else '1bp'
        assert self._embedding_mode in ['1bp', '128bp'], (
            f'Invalid embedding_mode: {self._embedding_mode}. '
            'Must be one of: "1bp", "128bp"'
        )
        
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict MPRA activity from embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings from AlphaGenome backbone.
                - embeddings.embeddings_1bp: (B, S, 1536) - from decoder (encoder+transformer)
                - embeddings.embeddings_128bp: (B, S//128, 3072) - from transformer
                - embeddings.embeddings_pair: (B, S//2048, S//2048, 128) - pairwise
            organism_index: Organism indices for each batch element.
            
        Returns:
            Per-position predictions array with shape (batch, sequence_length, num_tracks).
            This format is compatible with AlphaGenome's augmentation/reverse-complement logic.
        """
        # Get embeddings based on mode
        if self._embedding_mode == '1bp':
            # Use 1bp embeddings (has both local and global context)
            x = embeddings.get_sequence_embeddings(resolution=1)
            # x shape: (batch, sequence_length, 1536)
            
        elif self._embedding_mode == '128bp':
            # Use 128bp embeddings (pure transformer output, global context)
            x = embeddings.get_sequence_embeddings(resolution=128)
            # x shape: (batch, sequence_length//128, 3072)
        
        # Predict per-position scores
        hidden_dim = 256
        x = hk.Linear(hidden_dim, name='hidden')(x)
        x = jax.nn.relu(x)
        per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        # per_position_predictions shape: (batch, sequence_length, num_tracks)
        
        # Return per-position predictions directly (rank 3)
        # This is compatible with AlphaGenome's reverse_complement logic for RNA_SEQ
        return per_position_predictions
    
    def loss(self, predictions, batch):
        """Compute loss for MPRA predictions.
        
        Args:
            predictions: Per-position predictions with shape (batch, sequence_length, num_tracks)
                        where sequence_length depends on embedding_mode:
                        - 1bp mode: sequence_length in base pairs
                        - 128bp mode: sequence_length in 128bp bins
            batch: Batch data containing 'targets' with shape (batch,) or (batch, num_tracks)
        
        Returns:
            Dictionary with loss metrics
        """
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # predictions shape: (batch, sequence_length, num_tracks)
        # Pool over center region to get scalar predictions for loss computation.
        # IMPORTANT: All indexing values (seq_len, window_size, center_start) are kept
        # as plain Python integers so they are static and JIT-safe.
        seq_len = predictions.shape[1]  # Python int (static sequence length)
        
        # Convert center_bp to positions based on embedding resolution (Python ints only)
        if self._embedding_mode == '1bp':
            # For 1bp mode, center_bp is already in the right units
            window_size = int(self._center_bp)
        else:  # '128bp'
            # For 128bp mode, convert bp to positions (each position = 128bp)
            window_size = max(1, int(self._center_bp / 128))
        
        # For short sequences, use entire sequence (Python min to keep int)
        window_size = min(window_size, seq_len)
        center_start = (seq_len - window_size) // 2
        center_start = max(center_start, 0)
        
        # Use dynamic_slice_in_dim with static Python integer indices
        center_predictions = jax.lax.dynamic_slice_in_dim(
            predictions,
            start_index=center_start,
            slice_size=window_size,
            axis=1,
        )
        
        # Pool to get scalar per batch
        if self._pooling_type == 'mean':
            pred_values = jnp.mean(center_predictions, axis=1)  # (batch, num_tracks)
        elif self._pooling_type == 'max':
            pred_values = jnp.max(center_predictions, axis=1)  # (batch, num_tracks)
        elif self._pooling_type == 'sum':
            pred_values = jnp.sum(center_predictions, axis=1)  # (batch, num_tracks)
        
        if targets.ndim == 1:
            targets = targets[:, None]
        
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        # Calculate pearson correlation coefficient
        # Flatten to 1D for correlation calculation
        pred_flat = pred_values.flatten()
        targets_flat = targets.flatten()
        pearson_corr = jnp.corrcoef(pred_flat, targets_flat)[0, 1]
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
            'pearson_corr': pearson_corr,
        }
        
        

class EncoderMPRAHead(CustomHead):
    """MPRA head that uses raw encoder output (before transformer).
    
    This head REQUIRES that encoder_output is available in the embeddings object.
    It will fail if encoder_output is not provided - this is intentional to ensure
    you're actually using encoder features, not transformer-processed features.
    
    Characteristics:
    - Local convolutional features only (no global attention)
    - Operates at 128bp resolution (encoder output)
    - Different inductive bias (CNN vs attention)
    
    Requirements:
    - ExtendedEmbeddings with encoder_output field
    - Custom forward pass that captures encoder output
    
    Configuration:
    - center_bp: Center region to average over, in BASE PAIRS (default: 256bp)
                 This is automatically converted to encoder positions (128bp resolution).
                 Example: center_bp=384 → 384/128 = 3 encoder positions
    - pooling_type: How to pool over center region ('mean', 'sum', or 'max')
    """
    
    # Encoder resolution in base pairs per position
    ENCODER_RESOLUTION_BP = 128
    
    def __init__(
        self, 
        *, 
        name, 
        output_type, 
        num_tracks, 
        num_organisms, 
        metadata,
    ):
        super().__init__(
            name=name,
            num_tracks=num_tracks,
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )
        
        # Get center region in base pairs (default 256bp = 2 encoder positions)
        center_bp = metadata.get('center_bp', 256) if metadata else 256
        # Hidden layer sizes: can be int (single layer) or list (multiple layers)
        nl_size = metadata.get('nl_size', 1024) if metadata else 1024
        if isinstance(nl_size, int):
            self._hidden_sizes = [nl_size]
        elif isinstance(nl_size, list):
            self._hidden_sizes = nl_size
        else:
            raise ValueError(f"nl_size must be int or list, got {type(nl_size)}")
        do = metadata.get('do', None) if metadata else None
        self._do = do
        # Convert base pairs to encoder positions (128bp resolution)
        self._center_window_positions = max(1, int(center_bp / self.ENCODER_RESOLUTION_BP))
        
        pooling_type = (
            metadata.get('pooling_type', 'sum') if metadata else 'sum'
        )
        assert pooling_type in ['sum', 'mean', 'max', 'center', 'flatten'], \
            f"Invalid pooling type: {pooling_type}. Must be one of: sum, mean, max, center, flatten"
        self._pooling_type = pooling_type
        
        # Activation function
        activation = metadata.get('activation', 'relu') if metadata else 'relu'
        assert activation in ['relu', 'gelu'], f"Invalid activation: {activation}. Must be 'relu' or 'gelu'"
        self._activation = activation
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict using raw encoder output (before transformer).
        
        Returns:
            Per-position predictions array with shape (batch, sequence_length//128, num_tracks).
        
        Raises:
            AttributeError: If encoder_output is not available in embeddings.
            ValueError: If encoder_output is None.
        """
        # Require raw encoder output - no fallback!
        if not hasattr(embeddings, 'encoder_output'):
            raise AttributeError(
                "EncoderMPRAHead requires 'encoder_output' in embeddings object. "
                "Use MPRAHead with embedding_mode='128bp' if you want transformer features, "
                "or set use_encoder_output=True when creating the model."
            )
        
        if embeddings.encoder_output is None:
            raise ValueError(
                "encoder_output is None. Make sure the forward pass captures and "
                "provides the raw encoder output before the transformer."
            )
        
        # Use raw encoder output (BEFORE transformer)
        x = embeddings.encoder_output  # (batch, seq_len//128, D)
        # Prediction layers operating at 128bp resolution
        x = layers.LayerNorm(name='norm')(x)
        
        # Handle flatten pooling: flatten all positions before dense layers
        if self._pooling_type == 'flatten':
            # Flatten all positions: (batch, seq_len//128, D) -> (batch, seq_len//128 * D)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)  # Flatten spatial dimensions
            # Now x is (batch, flattened_features)
        # For other pooling types, keep per-position structure
        
        # Multiple hidden layers
        for i, hidden_size in enumerate(self._hidden_sizes):
            x = hk.Linear(hidden_size, name=f'hidden_{i}')(x)
            # Apply dropout only during training (when RNG is available)
            if self._do is not None:
                try:
                    rng_key = hk.next_rng_key()
                    x = hk.dropout(rng_key, self._do, x)
                except (RuntimeError, ValueError, AttributeError):
                    # RNG not available (evaluation mode) - skip dropout
                    pass
            # Apply activation
            if self._activation == 'gelu':
                x = jax.nn.gelu(x)
            else:  # relu
                x = jax.nn.relu(x)
        
        # Output layer
        if self._pooling_type == 'flatten':
            # For flatten, output is already per-sample: (batch, num_tracks)
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
            # Reshape to match expected format: (batch, 1, num_tracks) for consistency
            per_position_predictions = per_position_predictions[:, None, :]
        else:
            # For other pooling types, output per-position: (batch, seq_len//128, num_tracks)
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        # Return per-position predictions at 128bp resolution (rank 3)
        # Shape: (batch, seq_len//128, num_tracks) for normal pooling
        # Shape: (batch, 1, num_tracks) for flatten pooling
        return per_position_predictions
    
    def loss(self, predictions, batch):
        """Compute loss.
        
        Args:
            predictions: Per-position predictions with shape (batch, sequence_length, num_tracks)
            batch: Batch data containing 'targets' with shape (batch,) or (batch, num_tracks)
        
        Returns:
            Dictionary with loss metrics
        """
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # predictions shape: (batch, sequence_length_in_encoder_positions, num_tracks)
        # Pool over center region to get scalar predictions for loss computation.
        # Note: sequence_length is in encoder positions (128bp resolution).
        # IMPORTANT: Keep all indexing values as plain Python integers so they are
        # static when the loss is JIT-compiled in the cached-embedding (stage 1) path.
        seq_len = predictions.shape[1]  # Python int (length in encoder positions)
        
        # Handle different pooling types
        if self._pooling_type == 'flatten':
            # For flatten, predictions are already per-sample (batch, 1, num_tracks)
            # from the predict() function, so just squeeze the sequence dimension
            pred_values = predictions.squeeze(1)  # (batch, num_tracks)
        elif self._pooling_type == 'center':
            # Take only the center position.
            center_idx = seq_len // 2  # Python int
            center_predictions = jax.lax.dynamic_slice_in_dim(
                predictions,
                start_index=center_idx,
                slice_size=1,
                axis=1,
            )
            pred_values = center_predictions.squeeze(1)  # (batch, num_tracks)
        else:
            # For center window pooling (mean/max/sum)
            # Extract center window based on center_bp (converted to encoder positions)
            # IMPORTANT: use only Python ints here for JIT safety.
            window_size = min(int(self._center_window_positions), seq_len)
            center_start = (seq_len - window_size) // 2
            center_start = max(center_start, 0)
            
            # Extract center window using dynamic_slice_in_dim with static indices
            center_predictions = jax.lax.dynamic_slice_in_dim(
                predictions,
                start_index=center_start,
                slice_size=window_size,
                axis=1,
            )
            
            # Pool to get scalar per batch
            if self._pooling_type == 'mean':
                pred_values = jnp.mean(center_predictions, axis=1)  # (batch, num_tracks)
            elif self._pooling_type == 'max':
                pred_values = jnp.max(center_predictions, axis=1)  # (batch, num_tracks)
            elif self._pooling_type == 'sum':
                pred_values = jnp.sum(center_predictions, axis=1)  # (batch, num_tracks)
        if targets.ndim == 1:
            targets = targets[:, None]
        
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        # Calculate pearson correlation coefficient
        # Flatten to 1D for correlation calculation
        pred_flat = pred_values.flatten()
        targets_flat = targets.flatten()
        pearson_corr = jnp.corrcoef(pred_flat, targets_flat)[0, 1]
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
            'pearson_corr': pearson_corr,
        }


class DeepSTARRHead(CustomHead):
    """Head for DeepSTARR enhancer activity prediction with two outputs.
    
    Predicts two types of enhancer activity:
    - Developmental enhancer activity
    - Housekeeping enhancer activity
    
    Similar architecture to EncoderMPRAHead but with 2 output tracks.
    Uses raw encoder output (before transformer) at 128bp resolution.
    
    Configuration:
    - center_bp: Center region to pool over, in BASE PAIRS (default: 256bp)
    - pooling_type: How to pool ('mean', 'sum', 'max', 'center', 'flatten')
    - nl_size: Hidden layer size(s) - int or list
    - do: Dropout rate (optional)
    - activation: 'relu' or 'gelu'
    """
    
    ENCODER_RESOLUTION_BP = 128
    
    def __init__(
        self, 
        *, 
        name, 
        output_type, 
        num_tracks, 
        num_organisms, 
        metadata,
    ):
        super().__init__(
            name=name,
            num_tracks=num_tracks,  # Should be 2 for DeepSTARR
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )
        
        # Get configuration from metadata
        center_bp = metadata.get('center_bp', 256) if metadata else 256
        nl_size = metadata.get('nl_size', 1024) if metadata else 1024
        if isinstance(nl_size, int):
            self._hidden_sizes = [nl_size]
        elif isinstance(nl_size, list):
            self._hidden_sizes = nl_size
        else:
            raise ValueError(f"nl_size must be int or list, got {type(nl_size)}")
        
        self._do = metadata.get('do', None) if metadata else None
        self._center_window_positions = max(1, int(center_bp / self.ENCODER_RESOLUTION_BP))
        
        pooling_type = metadata.get('pooling_type', 'sum') if metadata else 'sum'
        assert pooling_type in ['sum', 'mean', 'max', 'center', 'flatten'], \
            f"Invalid pooling type: {pooling_type}"
        self._pooling_type = pooling_type
        
        activation = metadata.get('activation', 'relu') if metadata else 'relu'
        assert activation in ['relu', 'gelu'], f"Invalid activation: {activation}"
        self._activation = activation
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict using raw encoder output (before transformer).
        
        Returns:
            Per-position predictions with shape (batch, sequence_length//128, num_tracks=2).
        """
        # Require raw encoder output
        if not hasattr(embeddings, 'encoder_output'):
            raise AttributeError(
                "DeepSTARRHead requires 'encoder_output' in embeddings object. "
                "Set use_encoder_output=True when creating the model."
            )
        
        if embeddings.encoder_output is None:
            raise ValueError(
                "encoder_output is None. Make sure the forward pass captures the encoder output."
            )
        
        # Use raw encoder output
        x = embeddings.encoder_output  # (batch, seq_len//128, D)
        x = layers.LayerNorm(name='norm')(x)
        
        # Handle flatten pooling
        if self._pooling_type == 'flatten':
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # Multiple hidden layers
        for i, hidden_size in enumerate(self._hidden_sizes):
            x = hk.Linear(hidden_size, name=f'hidden_{i}')(x)
            # Apply dropout during training
            if self._do is not None:
                try:
                    rng_key = hk.next_rng_key()
                    x = hk.dropout(rng_key, self._do, x)
                except (RuntimeError, ValueError, AttributeError):
                    pass
            # Apply activation
            if self._activation == 'gelu':
                x = jax.nn.gelu(x)
            else:
                x = jax.nn.relu(x)
        
        # Output layer (2 tracks: developmental and housekeeping)
        if self._pooling_type == 'flatten':
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
            per_position_predictions = per_position_predictions[:, None, :]
        else:
            per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        return per_position_predictions
    
    def loss(self, predictions, batch):
        """Compute loss for DeepSTARR predictions (2 outputs).
        
        Args:
            predictions: Per-position predictions (batch, sequence_length, 2)
            batch: Batch data with 'targets' of shape (batch, 2)
                   targets[:, 0] = developmental activity
                   targets[:, 1] = housekeeping activity
        
        Returns:
            Dictionary with loss metrics including per-task Pearson correlations
        """
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # Use static Python integers for all indexing to keep JIT happy in both
        # cached-embedding (stage 1) and full-model (stage 2) training.
        seq_len = predictions.shape[1]  # Python int
        
        # Handle different pooling types
        if self._pooling_type == 'flatten':
            pred_values = predictions.squeeze(1)  # (batch, 2)
        elif self._pooling_type == 'center':
            center_idx = seq_len // 2  # Python int
            center_predictions = jax.lax.dynamic_slice_in_dim(
                predictions,
                start_index=center_idx,
                slice_size=1,
                axis=1,
            )
            pred_values = center_predictions.squeeze(1)  # (batch, 2)
        else:
            # For center window pooling (mean/max/sum)
            # Extract center window based on center_bp (converted to encoder positions)
            window_size = min(int(self._center_window_positions), seq_len)
            center_start = (seq_len - window_size) // 2
            center_start = max(center_start, 0)
            
            # Extract center window using dynamic_slice_in_dim with static indices
            center_predictions = jax.lax.dynamic_slice_in_dim(
                predictions,
                start_index=center_start,
                slice_size=window_size,
                axis=1,
            )
            
            if self._pooling_type == 'mean':
                pred_values = jnp.mean(center_predictions, axis=1)
            elif self._pooling_type == 'max':
                pred_values = jnp.max(center_predictions, axis=1)
            elif self._pooling_type == 'sum':
                pred_values = jnp.sum(center_predictions, axis=1)
        
        # Ensure targets are 2D
        if targets.ndim == 1:
            targets = targets[:, None]
        
        # Compute MSE loss
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        # Calculate overall Pearson correlation
        pred_flat = pred_values.flatten()
        targets_flat = targets.flatten()
        pearson_corr = jnp.corrcoef(pred_flat, targets_flat)[0, 1]
        
        # Calculate per-task Pearson correlations
        # Task 0: Developmental
        dev_pred = pred_values[:, 0]
        dev_target = targets[:, 0]
        dev_pearson = jnp.corrcoef(dev_pred, dev_target)[0, 1]
        
        # Task 1: Housekeeping
        hk_pred = pred_values[:, 1]
        hk_target = targets[:, 1]
        hk_pearson = jnp.corrcoef(hk_pred, hk_target)[0, 1]
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
            'pearson_corr': pearson_corr,
            'dev_pearson': dev_pearson,
            'hk_pearson': hk_pearson,
        }