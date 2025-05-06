"""
Projection layer for domain adaptation of financial embeddings.

This module provides a projection layer that transforms general-purpose embeddings
into a financial domain-specific embedding space.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import numpy as np

# Handle import compatibility based on installed versions
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.config.config import settings

logger = logging.getLogger(__name__)


class FinancialProjectionLayer:
    """
    Projection layer for financial domain adaptation.
    
    This layer transforms general embeddings into a financial-specific
    embedding space through a learned projection.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "tanh",
        dropout: float = 0.1
    ):
        """
        Initialize the financial projection layer.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
            hidden_dims: Dimensions of hidden layers (None for single layer)
            activation: Activation function to use
            dropout: Dropout rate
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or []
        self.activation_name = activation
        self.dropout_rate = dropout
        
        # Create numpy weights for fallback
        self.np_weights = None
        self.np_biases = None
        
        # Initialize the model if torch is available
        if HAS_TORCH:
            self._initialize_torch_model()
        else:
            self._initialize_numpy_fallback()
    
    def _initialize_torch_model(self):
        """Initialize PyTorch model."""
        # Define the layers
        layers = []
        
        # Input dimension
        current_dim = self.input_dim
        
        # Add hidden layers if specified
        if self.hidden_dims:
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(current_dim, hidden_dim))
                
                # Add activation
                if self.activation_name == "tanh":
                    layers.append(nn.Tanh())
                elif self.activation_name == "relu":
                    layers.append(nn.ReLU())
                elif self.activation_name == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.1))
                else:
                    layers.append(nn.ReLU())  # Default to ReLU
                
                # Add dropout
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                
                current_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        # Add final activation
        if self.activation_name == "tanh":
            layers.append(nn.Tanh())
        elif self.activation_name == "relu":
            layers.append(nn.ReLU())
        elif self.activation_name == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1))
            
        # Create the sequential model
        self.model = nn.Sequential(*layers)
        
        # Move model to available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize the weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights with Xavier initialization."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _initialize_numpy_fallback(self):
        """Initialize fallback numpy matrices."""
        # For simplicity, just create a random projection matrix for fallback
        np.random.seed(42)  # For reproducibility
        
        # Create weights
        self.np_weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
        
        # Create bias
        self.np_biases = np.zeros(self.output_dim)
        
        logger.warning("PyTorch not available, using numpy fallback for projection layer")
    
    def __call__(self, embeddings):
        """
        Apply the projection to the embeddings.
        
        Args:
            embeddings: Input embeddings (numpy array or torch tensor)
            
        Returns:
            Projected embeddings in the same format as input
        """
        if HAS_TORCH and isinstance(embeddings, torch.Tensor):
            return self.forward(embeddings)
        else:
            return self.forward_numpy(
                embeddings.numpy() if hasattr(embeddings, 'numpy') else embeddings
            )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PyTorch model.
        
        Args:
            embeddings: Input embeddings (torch tensor)
            
        Returns:
            Projected embeddings (torch tensor)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available for projection")
            
        # Move embeddings to the same device as the model
        embeddings = embeddings.to(self.device)
        
        # Apply the model
        return self.model(embeddings)
    
    def forward_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass using numpy for fallback.
        
        Args:
            embeddings: Input embeddings (numpy array)
            
        Returns:
            Projected embeddings (numpy array)
        """
        # Simple matrix multiplication for the projection
        projected = np.dot(embeddings, self.np_weights) + self.np_biases
        
        # Apply activation
        if self.activation_name == "tanh":
            projected = np.tanh(projected)
        elif self.activation_name == "relu":
            projected = np.maximum(0, projected)
        
        return projected
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the projection layer to the specified path.
        
        Args:
            path: Path to save the projection layer
        """
        if not HAS_TORCH:
            # Save numpy weights
            np.savez(
                path,
                weights=self.np_weights,
                biases=self.np_biases,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                activation=self.activation_name,
                dropout=self.dropout_rate
            )
            logger.info(f"Saved numpy projection layer to {path}")
            return
            
        # Create parent directory if it doesn't exist
        if isinstance(path, str):
            path = Path(path)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        if path.suffix != '.pt':
            path = path.with_suffix('.pt')
            
        # Save state dict and config
        state_dict = self.model.state_dict()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "dropout": self.dropout_rate
        }
        
        torch.save({
            "state_dict": state_dict,
            "config": config
        }, path)
        
        logger.info(f"Saved projection layer to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load the projection layer from the specified path.
        
        Args:
            path: Path to load the projection layer from
        """
        if isinstance(path, str):
            path = Path(path)
            
        if not path.exists():
            logger.warning(f"Projection layer file not found at {path}")
            return
            
        if not HAS_TORCH:
            # Load numpy weights
            try:
                data = np.load(path, allow_pickle=True)
                self.np_weights = data["weights"]
                self.np_biases = data["biases"]
                self.input_dim = int(data["input_dim"])
                self.output_dim = int(data["output_dim"])
                self.activation_name = str(data["activation"])
                self.dropout_rate = float(data["dropout"])
                logger.info(f"Loaded numpy projection layer from {path}")
            except Exception as e:
                logger.error(f"Error loading numpy projection layer: {e}")
            return
            
        # Load the model
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Get config
            config = checkpoint.get("config", {})
            
            # Update instance attributes if available
            self.input_dim = config.get("input_dim", self.input_dim)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.hidden_dims = config.get("hidden_dims", self.hidden_dims)
            self.activation_name = config.get("activation", self.activation_name)
            self.dropout_rate = config.get("dropout", self.dropout_rate)
            
            # Re-initialize the model with the loaded config
            self._initialize_torch_model()
            
            # Load the state dict
            state_dict = checkpoint.get("state_dict", checkpoint)  # Handle old format
            self.model.load_state_dict(state_dict)
            
            logger.info(f"Loaded projection layer from {path}")
        except Exception as e:
            logger.error(f"Error loading projection layer: {e}")
    
    @classmethod
    def from_pretrained(cls, path: Optional[Union[str, Path]] = None) -> "FinancialProjectionLayer":
        """
        Create a projection layer from a pretrained model.
        
        Args:
            path: Path to the pretrained model
            
        Returns:
            Pretrained projection layer
        """
        if path is None:
            path = settings.FINANCIAL_MODEL_DIR / "projection_layer.pt"
            
        if isinstance(path, str):
            path = Path(path)
            
        if not path.exists():
            logger.warning(f"Pretrained projection layer not found at {path}, initializing new layer")
            return cls()
            
        # Create a new instance
        instance = cls()
        
        # Load the pretrained weights
        instance.load(path)
        
        return instance
