"""Databricks-compatible training launcher for multi-GPU DDP."""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def launch_ddp_training(
    num_gpus: int = 8,
    config_path: str = 'configs/default_config.yaml',
    resume_checkpoint: str = None,
):
    """
    Launch DDP training on Databricks.
    
    Args:
        num_gpus: Number of GPUs to use
        config_path: Path to config file
        resume_checkpoint: Path to checkpoint to resume from
    """
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Build torchrun command
    cmd = [
        'torchrun',
        '--standalone',
        f'--nproc_per_node={num_gpus}',
        str(current_dir / 'train_ddp.py'),
        '--config', config_path,
    ]
    
    if resume_checkpoint is not None:
        cmd.extend(['--resume', resume_checkpoint])
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['NCCL_DEBUG'] = 'INFO'
    
    print("="*80)
    print(f"Launching DDP training with {num_gpus} GPUs on Databricks")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    # Run training
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"Training failed with return code {process.returncode}")
        print("="*80)
        raise RuntimeError(f"Training failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch DDP training on Databricks')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    launch_ddp_training(
        num_gpus=args.num_gpus,
        config_path=args.config,
        resume_checkpoint=args.resume,
    )

