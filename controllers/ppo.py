"""
PPO-based controller for the comma.ai controls challenge.
Loads a trained PPO model and uses it for inference.
"""
import numpy as np
from pathlib import Path
from typing import Optional

from . import BaseController

# Default path to trained model (update after training)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "best_model.zip"

# Constants (must match training environment)
FUTURE_HORIZON = 10


class Controller(BaseController):
    """
    PPO-based controller that uses a trained neural network policy.
    """
    def __init__(self, model_path: Optional[str] = None):
        # Lazy import to avoid loading torch/sb3 if not needed
        from stable_baselines3 import PPO
        
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"PPO model not found at {model_path}. "
                "Please train the model first using train_ppo.py"
            )
        
        self.model = PPO.load(model_path)
        print(f"Loaded PPO model from: {model_path}")
        
        # State tracking for observation construction
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
        """
        Build observation vector matching the training environment.
        IMPORTANT: Integral is updated BEFORE using (like PID).
        """
        error = target_lataccel - current_lataccel
        error_integral_after = self.error_integral + error  # Update FIRST
        error_derivative = error - self.prev_error
        
        # Future targets (pad if needed)
        future_targets = np.zeros(FUTURE_HORIZON, dtype=np.float32)
        if future_plan is not None and len(future_plan.lataccel) > 0:
            n = min(len(future_plan.lataccel), FUTURE_HORIZON)
            future_targets[:n] = future_plan.lataccel[:n]
        
        obs = np.array([
            target_lataccel,
            current_lataccel,
            error,
            state.roll_lataccel,
            state.v_ego / 30.0,  # normalized
            state.a_ego,
            self.prev_action,
            np.clip(error_integral_after, -10, 10),  # Updated integral
            error_derivative,
        ], dtype=np.float32)
        
        obs = np.concatenate([obs, future_targets])
        return obs
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Compute steering action using the trained PPO policy.
        """
        error = target_lataccel - current_lataccel
        
        # Build observation
        obs = self._build_observation(
            target_lataccel, current_lataccel, state, future_plan
        )
        
        # Get action from policy (deterministic for evaluation)
        action, _ = self.model.predict(obs, deterministic=True)
        action = float(action[0])
        
        # Clip action to valid range
        action = np.clip(action, -2.0, 2.0)
        
        # Update tracking state AFTER
        self.error_integral += error
        self.prev_error = error
        self.prev_action = action
        
        return action
