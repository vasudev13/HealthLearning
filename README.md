# HealthLearning

## Running Project on HPC

```
# Login to greene
ssh greene

# Switch to Log-4
ssh log-4

# Request CPU/GPU node
srun --account=csci_ga_2565_0001 --partition=interactive --pty /bin/bash (CPU)
                                            OR
srun --gres=gpu:1 --account=csci_ga_2565_0001 --partition=n1s8-t4-1 --time=04:00:00 --pty /bin/bash (GPU)

# Load singularity image and enviorment
singularity exec --nv --bind /scratch --bind /share/apps --overlay /share/apps/pytorch/1.8.1/pytorch-1.8.1.sqf:ro /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash

source /ext3/env.sh

# Assign a port
port=$(shuf -i 10000-65500 -n 1)

# Remember this port
echo $port

# Remote forward this port to greene
ssh -N -f -R $port:localhost:$port log-1.hpc.nyu.edu
ssh -N -f -R $port:localhost:$port log-2.hpc.nyu.edu
ssh -N -f -R $port:localhost:$port log-3.hpc.nyu.edu

# Start notebook on this port
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)

# On a separate terminal window
ssh -L <port>:localhost:<port> <net_id>@greene
```

---

## Perturbation Effects

### 1. Clinical Synonym Replacement

**1.1 MedNLI**

| Model        |  Accuracy | 
| ------------- |:-------------:|
| [Clinical BERT](https://arxiv.org/pdf/1904.03323.pdf) | 79.11 |
| [Clinical BERT](https://arxiv.org/pdf/1904.03323.pdf) + Clinical Synonym Replacement | 80.10 |
| Discharge Summary ALBERT | 78.13 |


