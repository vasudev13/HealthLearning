#!/bin/bash

#SBATCH --job-name=jupyter-gpu
#SBATCH --account=csci_ga_2565_0001
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue

module purge

port=$(shuf -i 10000-65500 -n 1)

user=$USER
/usr/bin/ssh -N -f -R $port:localhost:$port $user@log-1
/usr/bin/ssh -N -f -R $port:localhost:$port $user@log-2
/usr/bin/ssh -N -f -R $port:localhost:$port $user@log-3

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on greene compute node with above url

EOF

unset XDG_RUNTIME_DIR
unset SINGULARITY_BINDPATH
export SINGULARITY_CACHEDIR=/tmp/singularity-cache-${USER}

singularity exec --nv \
	    --bind /scratch \
	    --bind /share/apps \
	    --overlay /share/apps/pytorch/1.8.1/pytorch-1.8.1.sqf:ro \
            /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
jupyter lab --no-browser --port $port 
"
