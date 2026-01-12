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


class MPRAHead(CustomHead):
    """Custom head for predicting MPRA activity with flexible architecture access.
    
    Can use:
    - 1bp embeddings (encoder + transformer via decoder)
    - 128bp embeddings (pure transformer output)
    - Both combined for richer representations
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
        # Get center_window_size from metadata (default: 128 bp)
        self._center_window_size = (
            metadata.get('center_window_size', 128) if metadata else 128
        )
        # Get pooling_type from metadata (default: 'sum')
        pooling_type = (
            metadata.get('pooling_type', 'sum') if metadata else 'sum'
        )
        assert pooling_type in ['sum', 'mean', 'max'], f'Invalid pooling type: {pooling_type}'
        self._pooling_type = pooling_type
        
        # NEW: Choose which embeddings to use
        # Options: '1bp', '128bp', 'both'
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
            batch: Batch data containing 'targets' with shape (batch,) or (batch, num_tracks)
        
        Returns:
            Dictionary with loss metrics
        """
        targets = batch.get('targets')
        if targets is None:
            return {'loss': jnp.array(0.0)}
        
        # predictions shape: (batch, sequence_length, num_tracks)
        # Pool over center region to get scalar predictions for loss computation
        seq_len = predictions.shape[1]
        center_start = (seq_len - self._center_window_size) // 2
        center_start = jnp.maximum(center_start, 0)
        
        center_predictions = jax.lax.dynamic_slice_in_dim(
            predictions,
            start_index=center_start,
            slice_size=self._center_window_size,
            axis=1
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
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
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
        self._center_window_size = (
            metadata.get('center_window_size', 128) if metadata else 128
        )
        pooling_type = (
            metadata.get('pooling_type', 'sum') if metadata else 'sum'
        )
        assert pooling_type in ['sum', 'mean', 'max']
        self._pooling_type = pooling_type
    
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
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        
        # Return per-position predictions at 128bp resolution (rank 3)
        # Shape: (batch, seq_len//128, num_tracks)
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
        
        # predictions shape: (batch, sequence_length, num_tracks)
        # Pool over center region to get scalar predictions for loss computation
        seq_len = predictions.shape[1]
        center_start = (seq_len - self._center_window_size) // 2
        center_start = jnp.maximum(center_start, 0)
        
        center_predictions = jax.lax.dynamic_slice_in_dim(
            predictions,
            start_index=center_start,
            slice_size=self._center_window_size,
            axis=1
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
        return {'loss': mse_loss, 'mse': mse_loss}        