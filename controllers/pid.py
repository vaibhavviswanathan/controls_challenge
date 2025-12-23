from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # print(f"Target: {target_lataccdel}\nCurrent: {current_lataccel}\nState:{state}\nFuture:{future_plan}")
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    # print(f"Result: {self.p * error + self.i * self.error_integral + self.d * error_diff}\n\n")
    return self.p * error + self.i * self.error_integral + self.d * error_diff
