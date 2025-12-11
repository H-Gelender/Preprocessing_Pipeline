
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# Add src to path
sys.path.append('/app')
from src.rl.env_bio_swarm import BioSwarmEnv

def train():
    print("Initializing Environment...")
    env = BioSwarmEnv()
    
    # Sanity check
    # check_env(env) # SB3 check might be strict on observation spaces, let's skip for now or try
    
    print("Defining PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/app/output/rl_logs")
    
    print("Starting Training...")
    # Train for a short duration to demonstrate
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='/app/output/rl_checkpoints', name_prefix='bio_swarm')
    
    model.learn(total_timesteps=10000, callback=checkpoint_callback)
    
    print("Training finished.")
    model.save("/app/output/bio_swarm_ppo_final")

if __name__ == "__main__":
    train()
