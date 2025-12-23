"""
Gym environment wrapper for the TinyPhysics simulator.
Used for training RL agents (PPO) to control lateral acceleration.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator,
    CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH,
    STEER_RANGE, LATACCEL_RANGE, MAX_ACC_DELTA, DEL_T,
    LAT_ACCEL_COST_MULTIPLIER, State, FuturePlan
)
from controllers import BaseController


class DummyController(BaseController):
    """Placeholder controller - actions come from RL agent."""
    def __init__(self):
        self.next_action = 0.0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return self.next_action


class TinyPhysicsEnv(gym.Env):
    """
    OpenAI Gym environment for the comma.ai controls challenge.
    
    Observation space:
        - target_lataccel: target lateral acceleration
        - current_lataccel: current lateral acceleration
        - error: target - current
        - state: (roll_lataccel, v_ego, a_ego)
        - prev_action: previous steering command
        - error_integral: accumulated error (like PID I term)
        - error_derivative: change in error (like PID D term)
        - future_targets: next N target lataccel values
    
    Action space:
        - Continuous steering command in [-2, 2]
    
    Reward:
        - Negative of per-step cost (lataccel error + jerk penalty)
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Observation history length
    OBS_HISTORY = 5
    FUTURE_HORIZON = 10
    
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        reward_scale: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        self.data_files = sorted(list(self.data_dir.glob("*.csv")))
        assert len(self.data_files) > 0, f"No CSV files found in {data_dir}"
        
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        
        # Load physics model once
        self.physics_model = TinyPhysicsModel(model_path, debug=False)
        
        # Dummy controller - we'll override its action
        self.dummy_controller = DummyController()
        
        # Current simulator instance
        self.sim: Optional[TinyPhysicsSimulator] = None
        self.current_file_idx = 0
        
        # State tracking for observations
        self.prev_action = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0
        
        # Define observation space
        # [target, current, error, roll_lataccel, v_ego, a_ego, prev_action, 
        #  error_integral, error_derivative, future_targets(10)]
        obs_dim = 9 + self.FUTURE_HORIZON
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define action space: steering in [-2, 2]
        self.action_space = spaces.Box(
            low=np.array([STEER_RANGE[0]], dtype=np.float32),
            high=np.array([STEER_RANGE[1]], dtype=np.float32),
            dtype=np.float32
        )
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation from current simulator state.
        
        IMPORTANT: Must match how PID/BC see data.
        PID does error_integral += error BEFORE using it in the formula.
        """
        step_idx = self.sim.step_idx
        
        # Get current state info
        state, target, futureplan = self.sim.get_state_target_futureplan(step_idx)
        current = self.sim.current_lataccel
        error = target - current
        
        # Update integral FIRST (like PID does) for observation
        error_integral_after = self.error_integral + error
        
        # Error derivative
        error_derivative = error - self.prev_error
        
        # Future targets (pad if needed)
        future_targets = np.zeros(self.FUTURE_HORIZON, dtype=np.float32)
        if futureplan is not None and len(futureplan.lataccel) > 0:
            n = min(len(futureplan.lataccel), self.FUTURE_HORIZON)
            future_targets[:n] = futureplan.lataccel[:n]
        
        # Observation matching PID's internal view
        obs = np.array([
            target,                         # target lataccel
            current,                        # current lataccel
            error,                          # error
            state.roll_lataccel,           # roll contribution
            state.v_ego / 30.0,            # velocity (normalized ~highway speed)
            state.a_ego,                   # longitudinal accel
            self.prev_action,              # previous action
            np.clip(error_integral_after, -10, 10),  # integral term (updated FIRST)
            error_derivative,              # derivative term
        ], dtype=np.float32)
        
        obs = np.concatenate([obs, future_targets])
        return obs
    
    def _compute_reward(self, action: float) -> Tuple[float, Dict[str, float]]:
        """Compute reward based on lataccel error and jerk."""
        step_idx = self.sim.step_idx - 1  # step() increments it
        
        if step_idx < CONTROL_START_IDX:
            return 0.0, {"lataccel_cost": 0, "jerk_cost": 0}
        
        target = self.sim.target_lataccel_history[step_idx]
        current = self.sim.current_lataccel_history[step_idx]
        
        # Lataccel cost
        lataccel_cost = ((target - current) ** 2) * 100.0
        
        # Jerk cost
        if len(self.sim.current_lataccel_history) >= 2:
            prev_lataccel = self.sim.current_lataccel_history[step_idx - 1]
            jerk = (current - prev_lataccel) / DEL_T
            jerk_cost = (jerk ** 2) * 100.0
        else:
            jerk_cost = 0.0
        
        # Total cost (same as eval)
        total_cost = (lataccel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        
        # Reward is negative cost, scaled
        reward = -total_cost * self.reward_scale
        
        return reward, {
            "lataccel_cost": lataccel_cost,
            "jerk_cost": jerk_cost,
            "total_cost": total_cost
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Select a data file (cycle through or random)
        if options and "file_idx" in options:
            file_idx = options["file_idx"]
        else:
            file_idx = self.np_random.integers(0, len(self.data_files))
        
        self.current_file_idx = file_idx
        data_path = str(self.data_files[file_idx])
        
        # Create new simulator
        self.dummy_controller = DummyController()
        self.sim = TinyPhysicsSimulator(
            model=self.physics_model,
            data_path=data_path,
            controller=self.dummy_controller,
            debug=False
        )
        
        # Reset tracking state
        self.prev_action = 0.0
        self.prev_error = 0.0
        self.error_integral = 0.0
        
        obs = self._get_observation()
        info = {"data_file": data_path, "step_idx": self.sim.step_idx}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Extract scalar action
        act = float(np.clip(action[0], STEER_RANGE[0], STEER_RANGE[1]))
        
        # Update error tracking (for PID-like features)
        step_idx = self.sim.step_idx
        state, target, _ = self.sim.get_state_target_futureplan(step_idx)
        current = self.sim.current_lataccel
        error = target - current
        
        self.error_integral += error
        self.prev_error = error
        self.prev_action = act
        
        # Set action in dummy controller
        self.dummy_controller.next_action = act
        
        # Step simulation
        self.sim.step()
        
        # Compute reward
        reward, cost_info = self._compute_reward(act)
        
        # Check termination
        terminated = self.sim.step_idx >= len(self.sim.data)
        truncated = False
        
        # Get next observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "step_idx": self.sim.step_idx,
            **cost_info
        }
        
        # Add final cost if episode ended
        if terminated:
            final_cost = self.sim.compute_cost()
            info["final_cost"] = final_cost
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.sim.step_idx}: "
                  f"target={self.sim.target_lataccel_history[-1]:.2f}, "
                  f"current={self.sim.current_lataccel:.2f}, "
                  f"action={self.prev_action:.2f}")


class VecTinyPhysicsEnv:
    """
    Simple vectorized environment for faster training.
    Wraps multiple TinyPhysicsEnv instances.
    """
    def __init__(self, model_path: str, data_dir: str, num_envs: int = 4, **kwargs):
        self.envs = [
            TinyPhysicsEnv(model_path, data_dir, **kwargs) 
            for _ in range(num_envs)
        ]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
