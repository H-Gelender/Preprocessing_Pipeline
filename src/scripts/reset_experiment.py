
import shutil
import os
import sys

LOG_DIR = '/app/output/rl_logs'
CHECKPOINT_DIR = '/app/output/rl_checkpoints'

def reset():
    print("WARNING: This will delete all existing training logs and checkpoints.")
    confirm = input("Are you sure? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    if os.path.exists(LOG_DIR):
        print(f"Removing {LOG_DIR}...")
        shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR)
    
    if os.path.exists(CHECKPOINT_DIR):
        print(f"Removing {CHECKPOINT_DIR}...")
        shutil.rmtree(CHECKPOINT_DIR)
        os.makedirs(CHECKPOINT_DIR)
        
    print("\nReset complete.")
    print("To start a fresh training run (and watch progression):")
    print("  python /app/src/scripts/train_rl.py")
    print("\nThen refresh your Dashboard or generate a new report.")

if __name__ == "__main__":
    reset()
