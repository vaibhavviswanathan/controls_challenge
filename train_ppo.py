"""
PPO training script for the comma.ai controls challenge.
Uses stable-baselines3 for PPO implementation.
"""
import argparse
import os
import numpy as np
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from ppo_env import TinyPhysicsEnv


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


def train(args):
    """Main training loop."""
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_{timestamp}"
    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training PPO agent")
    print(f"  Model path: {args.model_path}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Log dir: {log_dir}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    
    # Create vectorized training environment
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(args.model_path, args.data_dir, i, args.seed)
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(args.model_path, args.data_dir, 0, args.seed)
        ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(args.model_path, args.data_dir, 0, args.seed + 1000)
    ])
    
    # PPO hyperparameters
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128]   # Value network
        ),
    )
    
    # Initialize PPO model
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=args.seed,
        )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,
        save_path=str(model_dir),
        name_prefix="ppo_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=args.eval_freq // args.num_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = model_dir / "ppo_final.zip"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for controls challenge")
    
    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to tinyphysics.onnx model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training CSV files")
    
    # Output paths
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--model_dir", type=str, default="./trained_models",
                        help="Directory to save model checkpoints")
    
    # Training settings
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per environment per update")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    
    # Callbacks
    parser.add_argument("--save_freq", type=int, default=50_000,
                        help="Checkpoint save frequency (timesteps)")
    parser.add_argument("--eval_freq", type=int, default=10_000,
                        help="Evaluation frequency (timesteps)")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
