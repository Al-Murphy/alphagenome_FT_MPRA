# Fine-tuning functionality for AlphaGenome

## API Reference

### Model Methods

All models (wrapped or with custom heads) support these methods:

#### Parameter Freezing

- **`freeze_parameters(freeze_paths=None, freeze_prefixes=None)`**  
  Freeze specific parameters by exact path or prefix.
  
- **`unfreeze_parameters(unfreeze_paths=None, unfreeze_prefixes=None)`**  
  Unfreeze specific parameters by exact path or prefix.
  
- **`freeze_backbone()`**  
  Freeze the backbone (encoder, transformer, decoder) but keep heads trainable.
  
- **`freeze_all_heads(except_heads=None)`**  
  Freeze all heads except those specified.
  
- **`freeze_except_head(trainable_head)`**  
  Freeze everything except a specific head (useful for finetuning).

#### Parameter Inspection

- **`get_parameter_paths()`**  
  Get list of all parameter paths in the model.
  
- **`get_head_parameter_paths()`**  
  Get list of all head parameter paths.
  
- **`get_backbone_parameter_paths()`**  
  Get list of all backbone parameter paths.
  
- **`count_parameters()`**  
  Count total number of parameters.

### Custom Head API

#### `CustomHead` Base Class

All custom heads must inherit from `CustomHead` and implement:

```python
class MyHead(CustomHead):
    def predict(self, embeddings, organism_index, **kwargs):
        """Generate predictions from embeddings.
        
        Args:
            embeddings: Multi-resolution embeddings from AlphaGenome backbone
            organism_index: Organism indices for each batch element
            
        Returns:
            Dictionary with predictions and any intermediate outputs
        """
        pass
    
    def loss(self, predictions, batch):
        """Compute loss for predictions.
        
        Args:
            predictions: Output from predict()
            batch: Training batch with targets
            
        Returns:
            Dictionary with 'loss' and any other metrics
        """
        pass
```

#### Head Registration

- **`register_custom_head(head_name, head_class, head_config)`**  
  Register a custom head for use in finetuning.

- **`is_custom_head(head_name)`**  
  Check if a head is registered.

- **`get_custom_head_config(head_name)`**  
  Get configuration for a registered head.

- **`list_custom_heads()`**  
  List all registered custom heads.

### Model Creation

- **`wrap_pretrained_model(base_model)`**  
  Wrap an existing AlphaGenomeModel to add parameter management methods.

- **`create_model_with_custom_heads(model_version, custom_heads, organism_settings=None, device=None)`**  
  Create a model with custom heads replacing standard heads.  
  Returns a model with pretrained backbone and your custom head(s).