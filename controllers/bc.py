"""
Pure Behavior Cloning controller for the comma.ai controls challenge.
Uses a neural network trained to imitate PID controller behavior.
No RL fine-tuning - just supervised learning on PID demonstrations.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

from . import BaseController

# Default path to trained BC model
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "bc_policy.pt"

# Observation constants (must match training)
FUTURE_HORIZON = 10
OBS_DIM = 9 + FUTURE_HORIZON


class BCPolicy(nn.Module):
    """Behavior cloning policy network."""
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 1, hidden_dims: list = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        
        self.features = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        features = self.features(obs)
        return self.mean(features)


class Controller(BaseController):
    """
    Pure Behavior Cloning controller.
    Imitates PID behavior using a neural network trained with supervised learning.
    """
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"BC model not found at {model_path}. "
                "Please train the model first using train_bc_ppo.py"
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = BCPolicy().to(self.device)
        self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy.eval()
        print(f"Loaded BC model from: {model_path}")
        
        # State tracking
        self.prev_action = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0
        
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_action = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0
    
    def _build_observation(
        self,
        target_lataccel: float,
        current_lataccel: float,
        state,
        future_plan,
    ) -> np.ndarray:
        """Build observation vector matching the training format.
        
        IMPORTANT: Must match how PID sees data during training.
        PID does error_integral += error BEFORE using it in the formula.
        """
        error = target_lataccel - current_lataccel
        # Update integral FIRST (like PID does)
        error_integral_after = self.error_integral + error
        error_derivative = error - self.prev_error
        
        future_targets = np.zeros(FUTURE_HORIZON, dtype=np.float32)
        if future_plan is not None and len(future_plan.lataccel) > 0:
            n = min(len(future_plan.lataccel), FUTURE_HORIZON)
            future_targets[:n] = future_plan.lataccel[:n]
        
        obs = np.array([
            target_lataccel,
            current_lataccel,
            error,
            state.roll_lataccel,
            state.v_ego / 30.0,
            state.a_ego,
            self.prev_action,
            np.clip(error_integral_after, -10, 10),
            error_derivative,
        ], dtype=np.float32)
        
        return np.concatenate([obs, future_targets])
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Compute steering action using the BC policy."""
        error = target_lataccel - current_lataccel
        
        obs = self._build_observation(
            target_lataccel, current_lataccel, state, future_plan
        )
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.policy(obs_tensor).item()
        
        action = np.clip(action, -2.0, 2.0)
        
        # Update tracking state AFTER (to match next iteration)
        self.error_integral += error
        self.prev_error = error
        self.prev_action = action
        
        return action
