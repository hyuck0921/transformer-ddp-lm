# Databricks Setup Guide

This guide explains how to run the Transformer Language Model on Databricks with multi-GPU DDP.

## Prerequisites

### Databricks Cluster Configuration

1. **Runtime**: 
   - DBR 13.3 ML or higher
   - Must have PyTorch 2.0+

2. **Instance Type (Driver)**:
   - For 8 GPUs:
     - `p3.16xlarge` (8x V100, 16GB each)
     - `p4d.24xlarge` (8x A100, 40GB each) 
     - `g5.48xlarge` (8x A10G, 24GB each)
   - For 4 GPUs:
     - `p3.8xlarge` (4x V100, 16GB each)
     - `g5.12xlarge` (4x A10G, 24GB each)
     - `g4dn.12xlarge` (4x T4, 16GB each)

3. **Workers**: 
   - Set to **0** (single-node multi-GPU training)

4. **Libraries**:
   - Will be installed via notebook (see Step 1)

## Important: No Additional Modules Required!

**PyTorch DDP works natively on Databricks!**

- No need for Horovod (unless using multi-node distributed training)
- No need for Spark ML
- Just PyTorch + torchrun

## Quick Start

### Option A: Using Notebook (Recommended)

1. **Upload Notebook**:
   - Import `notebooks/databricks_train.ipynb` to your Databricks workspace
   - Or create new notebook and copy cells

2. **Upload Project Files**:
   
   **Method 1: Git Clone**
   ```python
   %sh
   cd /dbfs/mnt/
   git clone https://github.com/your-username/transformer-ddp-lm.git
   ```
   
   **Method 2: DBFS Upload**
   ```python
   # Upload files to DBFS via Databricks UI
   # Data → DBFS → Upload
   ```
   
   **Method 3: Workspace Files**
   - Upload to `/Workspace/Users/<your-email>/transformer-ddp-lm/`

3. **Run the Notebook**:
   - Follow the notebook steps
   - It will handle everything automatically

### Option B: Command Line

If you prefer command-line:

```bash
# In Databricks notebook cell
%sh
cd /dbfs/mnt/transformer-ddp-lm
python databricks_train.py --num-gpus 8 --config configs/default_config.yaml
```

## Step-by-Step Setup

### 1. Create Cluster

In Databricks:
1. Go to **Compute** → **Create Cluster**
2. Select cluster name: `transformer-training`
3. Select **Databricks Runtime**: `13.3 LTS ML` or higher
4. Select **Worker type**: None (0 workers)
5. Select **Driver type**: 
   - For 8 GPUs: `p3.16xlarge` or `g5.48xlarge`
   - For 4 GPUs: `p3.8xlarge` or `g5.12xlarge`
6. Click **Create Cluster**

### 2. Install Dependencies

In notebook cell:
```python
%pip install einops pyyaml tensorboard tqdm
```

### 3. Upload Project

**Option 1: Git**
```python
%sh
cd /tmp
git clone https://github.com/your-username/transformer-ddp-lm.git
cd transformer-ddp-lm
```

**Option 2: DBFS**
```python
# Upload via UI first, then:
import os
os.chdir('/dbfs/mnt/transformer-ddp-lm')
```

### 4. Verify Setup

```python
!python test_setup.py
```

Should show:
- ✓ PyTorch installed
- ✓ 8 GPUs detected
- ✓ CUDA available
- ✓ NCCL backend available

### 5. Prepare Dataset

```python
!python data/prepare_dataset.py --output data/toy_dataset.txt --repeat 100
```

### 6. Train Model

**Single GPU Test (recommended first):**
```python
!python train_single.py --config configs/default_config.yaml
```

**Multi-GPU Training:**
```python
!python databricks_train.py --num-gpus 8 --config configs/default_config.yaml
```

Or directly:
```python
!torchrun --standalone --nproc_per_node=8 train_ddp.py --config configs/default_config.yaml
```

### 7. Monitor Training

**View logs:**
```python
!tail -f logs/train_*.log
```

**Check checkpoints:**
```python
!ls -lh checkpoints/
```

### 8. Inference

```python
!python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The Transformer" \
    --max-length 500
```

## File Paths on Databricks

### DBFS Paths
- Accessible from both driver and workers
- Persistent across cluster restarts
- Path format: `/dbfs/mnt/<path>` or `dbfs:/mnt/<path>`

### Workspace Files
- User-specific
- Path format: `/Workspace/Users/<email>/<path>`

### /tmp Directory
- Fast but temporary
- Deleted when cluster terminates
- Path format: `/tmp/<path>`

**Recommendation**: Use DBFS for persistence

## Common Issues & Solutions

### Issue 1: GPUs Not Detected
```python
import torch
print(f"GPUs: {torch.cuda.device_count()}")
```

**Solution**: 
- Verify cluster instance type has GPUs
- Restart cluster
- Check Databricks Runtime version (needs ML runtime)

### Issue 2: Out of Memory
**Solution**:
- Reduce `batch_size` in config (default: 32)
- Reduce model size (`dim`, `depth`)
- Enable gradient accumulation

### Issue 3: torchrun Not Found
**Solution**:
```python
# Use full path
!/databricks/python3/bin/torchrun --standalone --nproc_per_node=8 train_ddp.py
```

### Issue 4: Import Errors
**Solution**:
```python
# Make sure you're in project directory
import os
os.chdir('/dbfs/mnt/transformer-ddp-lm')

# Add to Python path
import sys
sys.path.insert(0, '/dbfs/mnt/transformer-ddp-lm')
```

### Issue 5: NCCL Errors
**Solution**:
```python
# Set environment variables
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
```

## Performance Tips

1. **Use Mixed Precision**
   - Set `use_amp: true` in config (default)
   - Reduces memory and speeds up training

2. **Optimize Batch Size**
   - Try batch sizes: 16, 32, 64 per GPU
   - Larger = faster but more memory

3. **Increase Workers**
   - Set `num_workers: 4-8` in config
   - Speeds up data loading

4. **Monitor GPU Utilization**
   ```python
   !watch -n 1 nvidia-smi
   ```
   - Should be >90% during training

## Cost Optimization

### Spot Instances
- Use spot instances for training (cheaper)
- Save checkpoints frequently
- Enable auto-resume

### Right-Sizing
- Start with 4 GPUs, scale to 8 if needed
- Use smaller instance for testing

### Auto-Termination
- Set cluster to auto-terminate after 30 min idle
- Save checkpoints to DBFS (persistent)

## Advanced Usage

### Resume Training
```python
!python databricks_train.py \
    --num-gpus 8 \
    --config configs/default_config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pt
```

### Custom Config
```python
import yaml

# Modify config
with open('configs/default_config.yaml') as f:
    config = yaml.safe_load(f)

config['training']['num_epochs'] = 200
config['training']['batch_size'] = 16

# Save
with open('configs/custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Train
!python databricks_train.py --config configs/custom_config.yaml
```

### Download Results
```python
# Copy to DBFS FileStore for easy download
import shutil
shutil.copytree(
    'checkpoints',
    '/dbfs/FileStore/transformer-lm/checkpoints'
)
```

Then download from:
```
https://<workspace>.cloud.databricks.com/files/transformer-lm/checkpoints/
```

## Comparison: Local vs Databricks

| Feature | Local Machine | Databricks |
|---------|--------------|------------|
| Setup | Manual | Managed |
| GPUs | Your hardware | Cloud instances |
| Scaling | Limited | Easy |
| Cost | Upfront | Pay-as-you-go |
| Collaboration | Difficult | Easy (notebooks) |
| Persistence | Local disk | DBFS |

## Next Steps

1. Run through `notebooks/databricks_train.ipynb`
2. Train with your own dataset
3. Experiment with model sizes
4. Share notebooks with team
5. Set up MLflow tracking (optional)

## Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Databricks ML Runtime](https://docs.databricks.com/runtime/mlruntime.html)
- [Databricks File System](https://docs.databricks.com/dbfs/index.html)

## Support

For issues:
1. Check this guide
2. Run `test_setup.py` for diagnostics
3. Check logs in `logs/` directory
4. Verify cluster configuration

