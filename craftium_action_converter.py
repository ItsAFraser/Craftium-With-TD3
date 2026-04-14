"""Utilities for mapping TD3 continuous outputs to Craftium discrete actions."""

import torch
import torch.nn as nn
import numpy as np


class ContinuousToDiscreteNN(nn.Module):
    """
    Neural network that converts continuous action values to discrete action logits.
    
    Input: 18 continuous values (one per Craftium action)
    Output: 18 logits representing the action probabilities via softmax
    
    This NN learns to map continuous TD3 outputs to the discrete action space
    through end-to-end training with policy gradients.
    """
    
    def __init__(self, num_actions=18, hidden_size=128, device="cpu"):
        """
        Initialize the converter network.
        
        Args:
            num_actions: Number of discrete actions (default 18 for OpenWorld)
            hidden_size: Size of hidden layers (default 128)
            device: PyTorch device ("cpu" or "cuda")
        """
        super(ContinuousToDiscreteNN, self).__init__()
        self.num_actions = num_actions
        self.device = device
        
        # Two-layer network: 18 -> 128 -> 128 -> 18 
        # TO Ben and Ethan, do you think this needs changed? i wondered if it was a bit excessive.
        self.net = nn.Sequential(
            nn.Linear(num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        self.to(device)
    
    def forward(self, continuous_actions):
        """
        Forward pass: map continuous actions to logits.
        
        Args:
            continuous_actions: Tensor of shape (batch_size, num_actions) or (num_actions,)
        
        Returns:
            logits: Tensor of shape (batch_size, num_actions) or (num_actions,)
        """
        return self.net(continuous_actions)
    
    def get_discrete_actions(self, continuous_actions, mode="argmax"):
        """
        Convert continuous actions to discrete action indices.
        
        Args:
            continuous_actions: numpy array or tensor of shape (batch_size, num_actions) or (num_actions,)
            mode: "argmax" for deterministic, "sample" for stochastic
        
        Returns:
            discrete_actions: numpy array of shape (batch_size,) or scalar
        """
        # Convert to tensor if needed
        if isinstance(continuous_actions, np.ndarray):
            continuous_actions = torch.from_numpy(continuous_actions).float().to(self.device)
        
        # Ensure 2D
        if continuous_actions.dim() == 1:
            continuous_actions = continuous_actions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            logits = self.forward(continuous_actions)
            probs = torch.softmax(logits, dim=-1)
            
            if mode == "argmax":
                actions = torch.argmax(probs, dim=-1)
            elif mode == "sample":
                actions = torch.multinomial(probs, 1).squeeze(-1)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        actions_np = actions.cpu().numpy()
        
        return actions_np[0] if squeeze_output else actions_np
    
    def get_action_probabilities(self, continuous_actions):
        """
        Get softmax probabilities for each action.
        
        Args:
            continuous_actions: numpy array or tensor
        
        Returns:
            probs: numpy array of shape (batch_size, num_actions) or (num_actions,)
        """
        if isinstance(continuous_actions, np.ndarray):
            continuous_actions = torch.from_numpy(continuous_actions).float().to(self.device)
        
        if continuous_actions.dim() == 1:
            continuous_actions = continuous_actions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            logits = self.forward(continuous_actions)
            probs = torch.softmax(logits, dim=-1)
        
        probs_np = probs.cpu().numpy()
        
        return probs_np[0] if squeeze_output else probs_np


def sample_gaussian_latent(mean, log_var, clamp_min=-6.0, clamp_max=2.0):
    """
    Sample z ~ N(mean, var) using reparameterization from numpy arrays.

    Args:
        mean: numpy array of means.
        log_var: numpy array of log-variances.
        clamp_min: lower bound for numerical stability.
        clamp_max: upper bound for numerical stability.

    Returns:
        numpy array sampled from the Gaussian distribution.
    """
    log_var = np.clip(log_var, clamp_min, clamp_max)
    std = np.exp(0.5 * log_var)
    eps = np.random.randn(*mean.shape).astype(np.float32)
    return mean + std * eps
