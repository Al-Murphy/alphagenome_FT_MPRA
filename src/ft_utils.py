"""
Utility functions for finetuning AlphaGenome models with custom heads.
"""

"""
MPRAHead that pools predictions over a center window.

This version:
1. Predicts per-position (per base-pair)
2. Extracts the center X bp (default 128 bp)
3. Pools predictions over that center region to get a single scalar score

# Example usage:
# 
# Option 1: Pass center_window_size through HeadConfig metadata
# register_custom_head(
#     'mpra_head',
#     MPRAHead,
#     HeadConfig(
#         type=HeadType.GENOME_TRACKS,
#         name='mpra_head',
#         output_type=dna_output.OutputType.RNA_SEQ,
#         num_tracks=1,
#         metadata={'center_window_size': 128},  # Pass as metadata
#     ),
# )
#
# Option 2: Pass it when creating the model (if you modify create_custom_head)
# For now, the easiest is to pass it in HeadConfig.metadata
"""

import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_ft import CustomHead


class MPRAHead(CustomHead):
    """Custom head for predicting MPRA (Massively Parallel Reporter Assay) activity.
    
    Predicts a single scalar score by:
    1. Predicting per-position scores
    2. Extracting the center window (default 128 bp)
    3. Pooling predictions over that center region
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
        # Get resolution from metadata (default: 128bp - output of the transformer tower)
        self._resolution = metadata.get('resolution', 128) if metadata else 128
    
    def predict(self, embeddings, organism_index, **kwargs):
        """Predict MPRA activity from embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings from AlphaGenome backbone.
            organism_index: Organism indices for each batch element.
            
        Returns:
            Dictionary with 'predictions' key containing a single scalar per sequence.
            Shape: (batch, 1)
        """
        # Get 1bp resolution embeddings
        x = embeddings.get_sequence_embeddings(resolution=self._resolution)
        # x shape: (batch, sequence_length, embed_dim)
        
        # Predict per-position scores
        x = hk.Linear(256, name='hidden')(x)
        x = jax.nn.relu(x)
        per_position_predictions = hk.Linear(self._num_tracks, name='output')(x)
        # per_position_predictions shape: (batch, sequence_length, num_tracks)
        
        # Extract center window
        seq_len = per_position_predictions.shape[1]
        
        # Calculate center region bounds
        # If sequence is shorter than window, use entire sequence
        window_size = jnp.minimum(seq_len, self._center_window_size)
        center_start = (seq_len - window_size) // 2
        center_end = center_start + window_size
        
        # Extract center region
        center_predictions = per_position_predictions[:, center_start:center_end, :]
        
        # Pool over center region to get single scalar
        # Options: mean, max, sum, etc.
        if self._pooling_type == 'mean':
            predictions = jnp.mean(center_predictions, axis=1)  # (batch, num_tracks)
        elif self._pooling_type == 'max':
            predictions = jnp.max(center_predictions, axis=1)  # (batch, num_tracks)
        elif self._pooling_type == 'sum':
            predictions = jnp.sum(center_predictions, axis=1)  # (batch, num_tracks)
        # If num_tracks=1, this gives (batch, 1)
        
        return {'predictions': predictions}
    
    def loss(self, predictions, batch):
        """Compute loss for MPRA predictions.
        
        Args:
            predictions: Output from predict(). Contains 'predictions' with shape (batch, 1).
            batch: Training batch with 'targets' key. Targets should be shape (batch,) or (batch, 1).
            
        Returns:
            Dictionary with 'loss' and any other metrics.
        """
        targets = batch.get('targets')
        if targets is None:
            # Return zero loss if no targets (e.g., during initialization)
            return {'loss': jnp.array(0.0)}
        
        pred_values = predictions['predictions']  # (batch, 1) or (batch, num_tracks)
        
        # Ensure targets have correct shape
        if targets.ndim == 1:
            targets = targets[:, None]  # (batch, 1)
        
        mse_loss = jnp.mean((pred_values - targets) ** 2)
        
        return {
            'loss': mse_loss,
            'mse': mse_loss,
        }

