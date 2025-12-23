# Controller Comparison Results

Comparison of different control approaches for the comma.ai controls challenge.

## Final Results (30 segments)

| Controller | Lataccel Cost | Jerk Cost | **Total Cost** |
|------------|---------------|-----------|----------------|
| **BC**     | 1.14          | 20.98     | **78.05**      |
| **PID**    | 1.15          | 20.88     | **78.30**      |
| PPO        | 3.24          | 16.35     | 178.19         |
| BC+PPO     | 6.39          | 13.26     | 332.75         |

## Controllers

### PID (Baseline)
- Simple PID controller with tuned gains: P=0.195, I=0.100, D=-0.053
- File: `controllers/pid.py`

### BC (Behavior Cloning)
- Neural network trained via supervised learning on PID demonstrations
- Architecture: 256 → 256 → 128 (ReLU activations)
- Trained on 50 episodes (~25k samples), 100 epochs
- **Matches PID performance exactly**
- File: `controllers/bc.py`
- Model: `trained_models/bc_policy.pt`

### PPO (Proximal Policy Optimization)
- RL agent trained from scratch using PPO
- 500k timesteps, 4 parallel environments
- File: `controllers/ppo.py`
- Model: `trained_models/ppo_v2/best_model.zip`

### BC+PPO (Behavior Cloning + PPO Fine-tuning)
- BC pre-training followed by PPO fine-tuning
- 500k PPO timesteps on top of BC initialization
- File: `controllers/bc_ppo.py`
- Model: `trained_models/bc_ppo_v2/bc_ppo_final.zip`

## Key Findings

1. **BC matches PID**: After fixing the integral tracking bug (integral must be updated BEFORE using, like PID does internally), behavior cloning successfully learns PID's policy with virtually identical performance.

2. **PPO from scratch underperforms**: Despite 500k steps of training, PPO achieves higher total cost than PID. It learns to reduce jerk but at the expense of tracking accuracy.

3. **BC+PPO degrades from BC**: PPO fine-tuning actually hurts performance, suggesting the RL optimization pushes away from the good BC initialization.

## Training Commands

```bash
# Train BC (included in BC+PPO script)
python train_bc_ppo.py --model_path ./models/tinyphysics.onnx --data_dir ./data --skip_bc

# Train PPO from scratch
python train_ppo.py --model_path ./models/tinyphysics.onnx --data_dir ./data \
    --total_timesteps 500000 --num_envs 4

# Train BC+PPO
python train_bc_ppo.py --model_path ./models/tinyphysics.onnx --data_dir ./data \
    --ppo_timesteps 500000 --num_envs 4
```

## Evaluation

```bash
# Evaluate any controller
python tinyphysics.py --model_path ./models/tinyphysics.onnx \
    --data_path ./data --controller <pid|bc|ppo|bc_ppo> --num_segs 100
```

## Important Implementation Note

The critical bug fix was ensuring the error integral is updated **BEFORE** using it in observations, matching how PID computes internally:

```python
# CORRECT (like PID)
error = target - current
error_integral += error  # Update FIRST
action = P * error + I * error_integral + D * error_derivative

# WRONG (causes distribution shift in BC)
error = target - current
action = P * error + I * error_integral + D * error_derivative  # Uses OLD integral
error_integral += error  # Update AFTER
```

## Future Improvements

To potentially beat PID with RL:
- Better reward shaping (e.g., reward for staying close to PID)
- Constrained PPO to prevent diverging from BC/PID
- Offline RL methods (CQL, IQL) that are more stable
- More training data diversity
