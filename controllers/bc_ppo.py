"""
Behavior Cloning + PPO controller for the comma.ai controls challenge.
Uses a model trained with BC pretraining followed by PPO fine-tuning.
"""
import numpy as np
from pathlib import Path
from typing import Optional

from . import BaseController

# Default path to trained model
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "trained_models" / "bc_ppo_final.zip"

# Observation constants (must match training)
FUTURE_HORIZON = 10


class Controller(BaseController):
    """
    BC+PPO controller that uses a neural network policy pretrained
    with behavior cloning on PID, then fine-tuned with PPO.
    """
    def __init__(self, model_path: Optional[str] = None):
        from stable_baselines3 import PPO
        
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
        
        # Also check for best_model from training
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            # Try best_model.zip in same directory
            best_model = model_path_obj.parent / "best_model.zip"
            if best_model.exists():
                model_path = str(best_model)
            else:
                raise FileNotFoundError(
                    f"BC+PPO model not found at {model_path}. "
                    "Please train the model first using train_bc_ppo.py"
                )
        
        self.model = PPO.load(model_path)
        print(f"Loaded BC+PPO model from: {model_path}")
        
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
        """Build observation vector matching the training environment.
        IMPORTANT: Integral is updated BEFORE using (like PID).
        """
        error = target_lataccel - current_lataccel
        error_integral_after = self.error_integral + error  # Update FIRST
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
            np.clip(error_integral_after, -10, 10),  # Updated integral
            error_derivative,
        ], dtype=np.float32)
        
        return np.concatenate([obs, future_targets])
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Compute steering action using the trained BC+PPO policy."""
        error = target_lataccel - current_lataccel
        
        obs = self._build_observation(
            target_lataccel, current_lataccel, state, future_plan
        )
        
        # Get action from policy (deterministic for evaluation)
        action, _ = self.model.predict(obs, deterministic=True)
        action = float(action[0])
        
        # Clip to valid range
        action = np.clip(action, -2.0, 2.0)
        
        # Update tracking state AFTER
        self.error_integral += error
        self.prev_error = error
        self.prev_action = action
        
        return action
