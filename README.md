# HealthLearning

## Running Project on HPC with Jupyter Notebook

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

## Language Modelling
```
  sbatch albert_train.sbatch
```

## Perturbation Effects

### 1. Clinical Synonym Replacement

**1.1 MedNLI**

| Model        |  Accuracy | 
| ------------- |:-------------:|
| [Clinical BERT](https://arxiv.org/pdf/1904.03323.pdf) | 79.11 (Reproduced) |
| [Clinical BERT](https://arxiv.org/pdf/1904.03323.pdf) + Clinical Synonym Replacement | 80.10 |
| Discharge Summary ALBERT | 78.13 |


## Usage
```
SCI_LG_transform=ClinicalSynonymSubstitution(substitution_probability=0.3,p=0.3,scispacy_entity_model="en_ner_bionlp13cg_md") 
BIONLP13CG_transform=ClinicalSynonymSubstitution(substitution_probability=0.7,p=0.7,scispacy_entity_model="en_ner_bionlp13cg_md")
BC5CDR_transform=ClinicalSynonymSubstitution(substitution_probability=0.7,p=0.7,scispacy_entity_model="en_ner_bionlp13cg_md")

composite_transform=Compose(
        [
            SCI_LG_transform, # For linking 2.78M unique concepts from UMLS, but not accurate yet, synonym candidates are generated without taking context information into account.
            BIONLP13CG_transform, # For cancer and genetics
            BC5CDR_transform # For chemicals and diseases
        ]
)

text="Patient has elevated BUN
composite_transform(text)

composite_transform=Compose(
        [
            SCI_LG_transform, # For linking 2.78M unique concepts from UMLS, but not accurate yet, synonym candidates are generated without taking context information into account.
            BIONLP13CG_transform, # For cancer and genetics
            BC5CDR_transform # For chemicals and diseases
        ]
)

text="Patient has elevated BUN
composite_transform(text)
```
