"""
Behavior Cloning + PPO training for the comma.ai controls challenge.
1. Collect expert demonstrations from PID controller
2. Train behavior cloning (BC) to imitate PID
3. Fine-tune with PPO
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ppo_env import TinyPhysicsEnv
from tinyphysics import (
    TinyPhysicsModel, TinyPhysicsSimulator, 
    CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE
)
from controllers.pid import Controller as PIDController


# Observation constants (must match ppo_env.py)
FUTURE_HORIZON = 10
OBS_DIM = 9 + FUTURE_HORIZON  # 19


def build_observation(target, current, state, future_plan, prev_action, error_integral, prev_error):
    """Build observation vector matching the PPO environment."""
    error = target - current
    error_derivative = error - prev_error
    
    future_targets = np.zeros(FUTURE_HORIZON, dtype=np.float32)
    if future_plan is not None and len(future_plan.lataccel) > 0:
        n = min(len(future_plan.lataccel), FUTURE_HORIZON)
        future_targets[:n] = future_plan.lataccel[:n]
    
    obs = np.array([
        target,
        current,
        error,
        state.roll_lataccel,
        state.v_ego / 30.0,
        state.a_ego,
        prev_action,
        np.clip(error_integral, -10, 10),
        error_derivative,
    ], dtype=np.float32)
    
    return np.concatenate([obs, future_targets])


def collect_pid_demonstrations(model_path: str, data_dir: str, num_episodes: int = 50):
    """Collect expert demonstrations from PID controller.
    
    IMPORTANT: We observe PID's internal state directly to avoid double-update bug.
    The simulator calls pid.update() in sim.step(), so we must NOT call it ourselves.
    """
    print(f"Collecting {num_episodes} episodes of PID demonstrations...")
    
    physics_model = TinyPhysicsModel(model_path, debug=False)
    data_files = sorted(list(Path(data_dir).glob("*.csv")))[:num_episodes]
    
    observations = []
    actions = []
    
    for data_file in tqdm(data_files, desc="Collecting demos"):
        pid = PIDController()
        sim = TinyPhysicsSimulator(physics_model, str(data_file), pid, debug=False)
        
        for step_idx in range(CONTEXT_LENGTH, len(sim.data)):
            state, target, futureplan = sim.get_state_target_futureplan(step_idx)
            current = sim.current_lataccel
            
            # Compute what PID will see BEFORE it updates
            # PID does: error_integral += error FIRST, then uses it
            error = target - current
            error_integral_after = pid.error_integral + error
            error_derivative = error - pid.prev_error
            prev_action = sim.action_history[-1] if sim.action_history else 0.0
            
            # Build observation matching what PID sees internally
            obs = build_observation(
                target, current, state, futureplan,
                prev_action, error_integral_after, pid.prev_error
            )
            
            # Let simulator step (this calls pid.update internally)
            sim.step()
            
            # Get the action that was actually taken
            action = sim.action_history[-1]
            
            # Store if in control region
            if step_idx >= CONTROL_START_IDX:
                observations.append(obs)
                actions.append(action)
    
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"Collected {len(observations)} observation-action pairs")
    return observations, actions


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
        mean = self.mean(features)
        return mean
    
    def get_distribution(self, obs):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)


def train_bc(observations: np.ndarray, actions: np.ndarray, 
             epochs: int = 100, batch_size: int = 256, lr: float = 1e-3,
             save_path: str = "bc_policy.pt"):
    """Train behavior cloning policy."""
    print(f"\nTraining BC policy for {epochs} epochs...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy = BCPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations).to(device)
    act_tensor = torch.FloatTensor(actions).to(device)
    
    dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_act in dataloader:
            optimizer.zero_grad()
            pred_act = policy(batch_obs)
            loss = criterion(pred_act, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), save_path)
    
    print(f"BC training complete. Best loss: {best_loss:.6f}")
    print(f"Saved BC policy to: {save_path}")
    return policy


def init_ppo_from_bc(bc_policy: BCPolicy, env, **ppo_kwargs):
    """Initialize PPO with weights from BC policy."""
    print("\nInitializing PPO from BC policy...")
    
    # Create PPO model
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        ),
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs
    )
    
    # Copy BC weights to PPO policy network
    bc_state = bc_policy.state_dict()
    ppo_policy = model.policy
    
    # Map BC layers to PPO policy layers
    # BC: features.0, features.2, features.4 -> PPO: mlp_extractor.policy_net.0, 2, 4
    with torch.no_grad():
        # Policy network (actor)
        ppo_policy.mlp_extractor.policy_net[0].weight.copy_(bc_state['features.0.weight'])
        ppo_policy.mlp_extractor.policy_net[0].bias.copy_(bc_state['features.0.bias'])
        ppo_policy.mlp_extractor.policy_net[2].weight.copy_(bc_state['features.2.weight'])
        ppo_policy.mlp_extractor.policy_net[2].bias.copy_(bc_state['features.2.bias'])
        ppo_policy.mlp_extractor.policy_net[4].weight.copy_(bc_state['features.4.weight'])
        ppo_policy.mlp_extractor.policy_net[4].bias.copy_(bc_state['features.4.bias'])
        
        # Action mean
        ppo_policy.action_net.weight.copy_(bc_state['mean.weight'])
        ppo_policy.action_net.bias.copy_(bc_state['mean.bias'])
        
        # Log std
        ppo_policy.log_std.copy_(bc_state['log_std'])
    
    print("Successfully initialized PPO from BC policy!")
    return model


def make_env(model_path: str, data_dir: str, rank: int, seed: int = 0):
    """Factory function to create environment instances."""
    def _init():
        env = TinyPhysicsEnv(
            model_path=model_path,
            data_dir=data_dir,
            reward_scale=0.01,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_ppo_finetune(model: PPO, env, eval_env, total_timesteps: int,
                       log_dir: str, model_dir: str):
    """Fine-tune PPO model."""
    print(f"\nFine-tuning with PPO for {total_timesteps} timesteps...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=model_dir,
        name_prefix="bc_ppo_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    return model


def main(args):
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bc_ppo_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect PID demonstrations
    if args.skip_bc and Path(args.bc_policy_path).exists():
        print(f"Loading existing BC policy from {args.bc_policy_path}")
        bc_policy = BCPolicy()
        bc_policy.load_state_dict(torch.load(args.bc_policy_path))
    else:
        observations, actions = collect_pid_demonstrations(
            args.model_path, args.data_dir, args.num_demo_episodes
        )
        
        # Step 2: Train BC
        bc_policy = train_bc(
            observations, actions,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            save_path=str(model_dir / "bc_policy.pt")
        )
    
    # Step 3: Create environments
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(args.model_path, args.data_dir, i, args.seed)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(args.model_path, args.data_dir, 0, args.seed)
        ])
    
    eval_env = DummyVecEnv([
        make_env(args.model_path, args.data_dir, 0, args.seed + 1000)
    ])
    
    # Step 4: Initialize PPO from BC
    ppo_model = init_ppo_from_bc(
        bc_policy, env,
        learning_rate=args.ppo_lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Lower entropy to stay closer to BC
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
    )
    
    # Step 5: Fine-tune with PPO
    ppo_model = train_ppo_finetune(
        ppo_model, env, eval_env,
        total_timesteps=args.ppo_timesteps,
        log_dir=str(log_dir),
        model_dir=str(model_dir),
    )
    
    # Save final model
    final_path = model_dir / "bc_ppo_final.zip"
    ppo_model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC + PPO training")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_dir", type=str, default="./trained_models")
    
    # BC settings
    parser.add_argument("--num_demo_episodes", type=int, default=50)
    parser.add_argument("--bc_epochs", type=int, default=100)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-3)
    parser.add_argument("--bc_policy_path", type=str, default="./trained_models/bc_policy.pt")
    parser.add_argument("--skip_bc", action="store_true", help="Skip BC, load existing")
    
    # PPO settings
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--ppo_timesteps", type=int, default=300000)
    parser.add_argument("--ppo_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
