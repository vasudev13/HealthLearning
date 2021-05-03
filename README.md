# HealthLearning

### Running Project on HPC

```
ssh greene
srun --account=csci_ga_2565_0001 --partition=interactive --pty /bin/bash
singularity exec --nv --bind /scratch --bind /share/apps --overlay /share/apps/pytorch/1.8.1/pytorch-1.8.1.sqf:ro /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash
source /ext3/env.sh
# assign a port
port=$(shuf -i 10000-65500 -n 1)

# remember this port
echo $port

# Remote forward this port to greene
ssh -N -f -R $port:localhost:$port log-1.hpc.nyu.edu
ssh -N -f -R $port:localhost:$port log-2.hpc.nyu.edu
ssh -N -f -R $port:localhost:$port log-3.hpc.nyu.edu

# start notebook on this port
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
ssh -L <port>:localhost:<port> <net_id>@greene
```