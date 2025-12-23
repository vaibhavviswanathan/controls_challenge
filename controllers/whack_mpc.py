from . import BaseController
import numpy as np
from tinyphysics import TinyPhysicsModel

MODEL_PATH = "/Users/vaibhavviswanathan/projects/controls_challenge/models/tinyphysics.onnx"

MPC_HORIZON = 5
MPC_NUM_CANDIDATES = 101
MPC_ACTION_RANGE = 0.15   # Smaller range to stay closer to PID
MPC_ACTION_DELTA_COST = 2.0 #1.0
MPC_DEBUG = True  # Set to True to print debug info

# Use a separate RNG for MPC to avoid polluting the simulator's RNG
MPC_RNG = np.random.default_rng(42)

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    # PID parameters
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0

    # also use MPC-ish
    self.tpm = TinyPhysicsModel(MODEL_PATH, debug=False)

    self.state_history = []
    self.action_history = []
    self.current_lataccel_history = []
    self.reset()

  def _estimate_step_idx(self) -> int:
    return len(self.action_history) + CONTEXT_LENGTH

  def _get_horizon(self, future_plan) -> int:
    if future_plan is None:
      return 1
    min_len = min(
      len(future_plan.lataccel),
      len(future_plan.roll_lataccel),
      len(future_plan.v_ego),
      len(future_plan.a_ego),
    )
    return int(max(1, min(MPC_HORIZON, min_len + 1)))

  def reset(self):
    self.state_history = []
    self.action_history = []
    self.current_lataccel_history = []

    self.error_integral = 0
    self.prev_error = 0

    # idk why
    # seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    # np.random.seed(seed)


  def record(self, step_idx, state, future_plan, target_lataccel, action, current_lataccel):
    self.state_history.append(state)
    self.action_history.append(action)
    self.current_lataccel_history.append(current_lataccel)


  def calc_pid(self, target_lataccel, current_lataccel, state, future_plan):

    # print(f"Target: {target_lataccel}\nCurrent: {current_lataccel}") #\nState:{state}\nFuture:{future_plan}")
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + self.d * error_diff
  

  def _predict_single_step(self, actions: np.ndarray, current_lataccel: float, state) -> np.ndarray:
    """Predict lataccel for given actions (for debugging)."""
    batch = len(actions)
    
    past_actions = np.asarray(self.action_history[-(CONTEXT_LENGTH - 1):], dtype=np.float32)
    past_states = np.asarray([list(s) for s in self.state_history[-(CONTEXT_LENGTH - 1):]], dtype=np.float32)
    past_preds = np.asarray(self.current_lataccel_history[-CONTEXT_LENGTH:], dtype=np.float32)

    current_state = np.asarray(list(state), dtype=np.float32)
    state_ctx_full = np.concatenate([past_states, current_state[None, :]], axis=0)

    state_ctx = np.tile(state_ctx_full[None, :, :], (batch, 1, 1))
    actions_ctx = np.tile(np.concatenate([past_actions, np.zeros((1,), dtype=np.float32)], axis=0)[None, :], (batch, 1))
    preds_ctx = np.tile(past_preds[None, :], (batch, 1))

    actions_ctx[:, -1] = actions
    model_states = np.concatenate([actions_ctx[:, :, None], state_ctx], axis=2)
    pred = self.tpm.get_current_lataccel_deterministic_batch(model_states, preds_ctx)
    pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
    return pred

  def _rollout_cost_1d(self, first_actions: np.ndarray, target_lataccel: float, current_lataccel: float, state, future_plan, horizon: int, debug: bool = False) -> np.ndarray:
    """
    1D Shooting MPC: evaluate candidate first actions.
    For each candidate, assume constant action over the horizon.
    """
    batch = first_actions.shape[0]

    if len(self.action_history) < CONTEXT_LENGTH:
      return np.full((batch,), np.inf)
    if len(self.state_history) < CONTEXT_LENGTH:
      return np.full((batch,), np.inf)
    if len(self.current_lataccel_history) < CONTEXT_LENGTH:
      return np.full((batch,), np.inf)

    # Controller histories have steps [0..T-1], but simulator uses [T-19..T]
    # So we take 19 from history + current state/candidate action
    past_actions = np.asarray(self.action_history[-(CONTEXT_LENGTH - 1):], dtype=np.float32)  # 19 actions
    past_states = np.asarray([list(s) for s in self.state_history[-(CONTEXT_LENGTH - 1):]], dtype=np.float32)  # 19 states
    past_preds = np.asarray(self.current_lataccel_history[-CONTEXT_LENGTH:], dtype=np.float32)  # 20 preds (this one matches)

    current_state = np.asarray(list(state), dtype=np.float32)
    state_ctx_full = np.concatenate([past_states, current_state[None, :]], axis=0)  # 20 states

    state_ctx = np.tile(state_ctx_full[None, :, :], (batch, 1, 1))
    # 19 past actions + placeholder for candidate = 20 actions
    actions_ctx = np.tile(np.concatenate([past_actions, np.zeros((1,), dtype=np.float32)], axis=0)[None, :], (batch, 1))
    preds_ctx = np.tile(past_preds[None, :], (batch, 1))
    
    if debug:
      print(f"MPC context shapes: actions={actions_ctx.shape}, states={state_ctx.shape}, preds={preds_ctx.shape}")
      print(f"MPC past_actions[-3:]: {past_actions[-3:]}")
      print(f"MPC current_state: {current_state}")
      print(f"MPC past_preds[-3:]: {past_preds[-3:]}")

    u_prev = float(self.action_history[-1])
    prev_lat = np.full((batch,), float(current_lataccel), dtype=np.float32)
    total_cost = np.zeros((batch,), dtype=np.float32)

    for k in range(horizon):
      # 1D shooting: use the same first_action for all steps (constant action assumption)
      actions_ctx[:, -1] = first_actions
      model_states = np.concatenate([actions_ctx[:, :, None], state_ctx], axis=2)
      pred = self.tpm.get_current_lataccel_deterministic_batch(model_states, preds_ctx)
      pred = np.clip(pred, prev_lat - MAX_ACC_DELTA, prev_lat + MAX_ACC_DELTA)

      if k == 0:
        ref = float(target_lataccel)
      else:
        ref = float(future_plan.lataccel[k - 1])

      lat_cost = ((ref - pred) ** 2) * 100.0
      jerk_cost = (((pred - prev_lat) / DEL_T) ** 2) * 100.0
      step_cost = (lat_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

      # Action delta cost only on first step (comparing to previous applied action)
      if k == 0:
        du = first_actions - u_prev
        step_cost = step_cost + (MPC_ACTION_DELTA_COST * (du ** 2) * 100.0)

      total_cost = total_cost + step_cost
      prev_lat = pred.astype(np.float32)

      if k < horizon - 1:
        preds_ctx[:, :-1] = preds_ctx[:, 1:]
        preds_ctx[:, -1] = prev_lat

        actions_ctx[:, :-1] = actions_ctx[:, 1:]

        if k == 0:
          next_state = np.array([
            future_plan.roll_lataccel[0],
            future_plan.v_ego[0],
            future_plan.a_ego[0],
          ], dtype=np.float32)
        else:
          next_state = np.array([
            future_plan.roll_lataccel[k],
            future_plan.v_ego[k],
            future_plan.a_ego[k],
          ], dtype=np.float32)

        state_ctx[:, :-1, :] = state_ctx[:, 1:, :]
        state_ctx[:, -1, :] = next_state[None, :]

    return total_cost


  def calc_mpc(self, target_lataccel, current_lataccel, state, future_plan, pid_result):
    """
    1D Shooting MPC: grid search over first action, assume constant action over horizon.
    """
    horizon = self._get_horizon(future_plan)
    
    # Generate candidate first actions: grid around PID result
    candidates = np.linspace(
      max(STEER_RANGE[0], pid_result - MPC_ACTION_RANGE),
      min(STEER_RANGE[1], pid_result + MPC_ACTION_RANGE),
      MPC_NUM_CANDIDATES
    ).astype(np.float32)

    costs = self._rollout_cost_1d(candidates, target_lataccel, current_lataccel, state, future_plan, horizon)
    best_idx = np.argmin(costs)
    best_action = float(candidates[best_idx])
    
    if MPC_DEBUG:
      pid_idx = np.argmin(np.abs(candidates - pid_result))
      min_cost_idx = np.argmin(costs)
      max_cost_idx = np.argmax(costs)
      print(f"err={target_lataccel-current_lataccel:+.2f} | PID={pid_result:+.3f}(cost={costs[pid_idx]:.0f}) | MPC={best_action:+.3f}(cost={costs[best_idx]:.0f}) | range=[{candidates[0]:+.2f},{candidates[-1]:+.2f}] | costs=[{costs[min_cost_idx]:.0f},{costs[max_cost_idx]:.0f}]")
      # Print which action has min cost vs PID
      if abs(best_action - pid_result) > 0.05:
        print(f"  -> MPC differs from PID by {best_action-pid_result:+.3f}. Best action {best_action:.3f}, PID action {pid_result:.3f}")
        # Debug: show what model predicts for a few actions
        test_actions = np.array([candidates[0], pid_result, best_action, candidates[-1]], dtype=np.float32)
        test_preds = self._predict_single_step(test_actions, current_lataccel, state)
        print(f"  -> Model predicts: action={candidates[0]:.2f}->lat={test_preds[0]:.3f}, action={pid_result:.2f}->lat={test_preds[1]:.3f}, action={best_action:.2f}->lat={test_preds[2]:.3f}, action={candidates[-1]:.2f}->lat={test_preds[3]:.3f}")
    
    return best_action


  def update(self, target_lataccel, current_lataccel, state, future_plan):
    action = self.calc_pid(
      target_lataccel=target_lataccel, 
      current_lataccel=current_lataccel,
      state=state,
      future_plan=future_plan
    )

    step_idx = self._estimate_step_idx()
    if step_idx >= CONTROL_START_IDX and future_plan is not None:
      if len(self.action_history) >= CONTEXT_LENGTH and len(self.current_lataccel_history) >= CONTEXT_LENGTH and len(self.state_history) >= CONTEXT_LENGTH:
        action = self.calc_mpc(
          target_lataccel=target_lataccel,
          current_lataccel=current_lataccel,
          state=state,
          future_plan=future_plan,
          pid_result=action,
        )

    action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))

    return action

