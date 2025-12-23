from . import BaseController
import numpy as np
from tinyphysics import TinyPhysicsModel, LataccelTokenizer
import copy

MODEL_PATH = "/Users/vaibhavviswanathan/projects/controls_challenge/models/tinyphysics.onnx"

MPC_RANGE = 0.1
MPC_BINS = 11 #101

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

  def reset(self):
    self.state_history = []
    self.action_history = []
    self.current_lataccel_history = []

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
  

  def calc_cost(self, target_lataccel, next_lataccel, prev_lataccel):
    lat_accel_cost = ((target_lataccel - next_lataccel)**2) * 100
    if prev_lataccel is None:
      jerk_cost = 0.0
    else:
      jerk_cost = (((next_lataccel - prev_lataccel) / DEL_T)**2) * 100

    return (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost

  def predict_and_calc_cost(self, target_lataccel, current_lataccel, state, action,):

    # temporary histories
    state_history=copy.deepcopy(self.state_history) + [state]
    action_history=copy.deepcopy(self.action_history) + [action]
    current_lataccel_history=copy.deepcopy(self.current_lataccel_history)

    # copied from sim_step
    pred = self.tpm.get_current_lataccel_deterministic(
      sim_states=state_history[-CONTEXT_LENGTH:],
      actions=action_history[-CONTEXT_LENGTH:],
      past_preds=current_lataccel_history[-CONTEXT_LENGTH:]
    )

    pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
    prev_lataccel = current_lataccel
    current_lataccel = pred
    
    # TODO: something weird with lat_accel
    return current_lataccel, self.calc_cost(target_lataccel, current_lataccel, prev_lataccel)

  
  def calc_mpc(self, target_lataccel, current_lataccel, state, future_plan, pid_result):
    print("\n\n\n=====---====")
    best_action = pid_result
    best_cost = 0
    best_lataccel, best_cost = self.predict_and_calc_cost(
      target_lataccel=target_lataccel,
      current_lataccel=current_lataccel,
      state = state,
      action=best_action,
    )

    for candidate_action in np.linspace(pid_result - MPC_RANGE, pid_result + MPC_RANGE, MPC_BINS):
      # Run sim & calc cost
      candidate_lataccel, candidate_cost = self.predict_and_calc_cost(
        target_lataccel=target_lataccel,
        current_lataccel=current_lataccel,
        state = state,
        action=candidate_action,
      )
      print(f"Canddiate action|cost: {candidate_action} . {pid_result} | {candidate_cost} . {best_cost}")

      # update candidate
      if candidate_cost < best_cost:
        best_cost = candidate_cost
        best_action = candidate_action
        best_lataccel = candidate_lataccel

    return best_action, best_lataccel


  def update(self, target_lataccel, current_lataccel, state, future_plan):
    action = self.calc_pid(
      target_lataccel=target_lataccel, 
      current_lataccel=current_lataccel,
      state=state,
      future_plan=future_plan
    )

    # do not run MPC until full context is bult
    lataccel = current_lataccel
    if len(self.action_history) >= CONTEXT_LENGTH:
      action, lataccel = self.calc_mpc(
        target_lataccel=target_lataccel, 
        current_lataccel=current_lataccel,
        state=state,
        future_plan=future_plan,
        pid_result=action,
      )

    return action

