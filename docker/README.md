# Docker images

Two images, built from `docker/Dockerfile` (targets of one file, sharing the system layers):

| target | numpy | architectures | for |
|---|---|---|---|
| `main` | 2 | x86_64, aarch64 | everything except RoboFactory's expert: all environments (MultiGrid, RWARE, VMAS, RoboFactory), data collection with the other experts and random policies, training (torch and JAX on CUDA: LeWM, the gcrl baselines, MAMBA), evaluation, notebooks (nbdev) |
| `robofactory-data` | < 2 | x86_64 | RoboFactory's motion-planning expert, i.e. `scripts/data/collect_robofactory.py`: its planner (mplib 0.1.1, pinned by ManiSkill) crashes with numpy 2, and JAX needs numpy 2, so they cannot share an environment |

`main` is one tag for both architectures: Docker and Apptainer pick the build of the machine they run on.
**The aarch64 build has no RoboFactory** (no ManiSkill): SAPIEN, its simulator, and mplib publish no aarch64
builds. Everything else (MultiGrid, RWARE, VMAS, training, evaluation) is the same.

Both have the repository at `/opt/stable-marl` (installed in editable mode); the x86_64 builds also have
RoboFactory's 3D assets (`/opt/robofactory/assets`, so compute nodes need no internet) and render ManiSkill
scenes on NVIDIA GPUs (Vulkan) or, without one, on the CPU (Mesa's lavapipe, slower).

## Prebuilt images (GitHub Container Registry)

The `Docker images` workflow (`.github/workflows/docker.yml`) builds both images at every push to master
that changes the code, checks them (imports, the environment examples, RoboFactory's expert on the CPU
renderer) and pushes them as `ghcr.io/ahmed-khaled-saleh/stable-marl:main` and `:robofactory-data` (plus
`:main-<commit>` / `:robofactory-data-<commit>` to pin a version; `:main-amd64` / `:main-arm64` are the
single-architecture builds behind `:main`). Run it by hand from the Actions tab
(*Run workflow*).

```bash
docker pull ghcr.io/ahmed-khaled-saleh/stable-marl:main
apptainer pull stable-marl-main.sif docker://ghcr.io/ahmed-khaled-saleh/stable-marl:main
apptainer pull stable-marl-rf-data.sif docker://ghcr.io/ahmed-khaled-saleh/stable-marl:robofactory-data
```

New GHCR packages are private: either make them public (GitHub -> your profile -> Packages ->
`stable-marl` -> Package settings -> Change visibility), or log in first with a personal access token
with the `read:packages` scope (`docker login ghcr.io` / `apptainer remote login --username <user> docker://ghcr.io`).

## Build

To build them yourself, from the repository root:

```bash
docker build -f docker/Dockerfile --target main -t stable-marl:main .
docker build -f docker/Dockerfile --target robofactory-data -t stable-marl:robofactory-data .
# JAX for CUDA 13 instead of 12 (match the host driver):
docker build -f docker/Dockerfile --target main --build-arg JAX_CUDA=cuda13 -t stable-marl:main .
```

`main` is several GB (CUDA libraries of torch and JAX); `robofactory-data` uses CPU torch and is smaller.

## Run with Docker

Datasets and checkpoints go to `$STABLEMARL_HOME` (default `~/.stable_marl`); mount a host folder there:

```bash
# RoboFactory expert data (numpy < 2)
docker run --rm --gpus all -e STABLEMARL_HOME=/data -v $PWD/data:/data stable-marl:robofactory-data \
    python scripts/data/collect_robofactory.py env.task=LiftBarrier expert_episodes=150

# training and evaluation on it (numpy 2)
docker run --rm --gpus all -e STABLEMARL_HOME=/data -v $PWD/data:/data stable-marl:main \
    python scripts/train/gcrl.py ...
```

To run your working copy instead of the code baked into the image, mount it over the installed one:
`-v $PWD:/opt/stable-marl`. `--gpus all` needs the NVIDIA Container Toolkit; leave it out on CPU machines.

## Run on an HPC cluster (Apptainer / Singularity)

Clusters usually run Apptainer rather than Docker. Convert the images once (on a machine with Docker, or
from a registry you pushed them to):

```bash
apptainer build stable-marl-main.sif docker-daemon://stable-marl:main
apptainer build stable-marl-rf-data.sif docker-daemon://stable-marl:robofactory-data
# or: docker push <registry>/stable-marl:main, then apptainer build stable-marl-main.sif docker://<registry>/stable-marl:main
```

and run with `--nv` (GPU drivers from the host) from the repository checkout:

```bash
apptainer exec --nv --env STABLEMARL_HOME=$SCRATCH/stable_marl stable-marl-rf-data.sif \
    python scripts/data/collect_robofactory.py env.task=LiftBarrier shard=$SLURM_ARRAY_TASK_ID
apptainer exec --nv --env STABLEMARL_HOME=$SCRATCH/stable_marl stable-marl-main.sif \
    python -c "import stable_marl.data as d; d.merge([...], 'robofactory_liftbarrier.h5')"
```

Apptainer mounts the current directory and your home, so the scripts of your checkout run (with the
package installed in the image: add `--env PYTHONPATH=$PWD` to use the checkout's `stable_marl` too).

## Roihu (CSC)

Roihu has two architectures: the CPU nodes are x86_64 (AMD Turin, login `roihu-cpu.csc.fi`) and the GPU
nodes aarch64 (NVIDIA GH200, login `roihu-gpu.csc.fi`). Build each `.sif` on the login node of the side it
will run on (as CSC recommends), so Apptainer takes the matching build of `:main`. Images take several GB:
keep Apptainer's cache out of the 15 GiB home quota.

```bash
export APPTAINER_CACHEDIR=/scratch/<project>/$USER/.apptainer
# on roihu-cpu.csc.fi: RoboFactory data (and evaluation in RoboFactory envs)
apptainer build --bind="$TMPDIR:/tmp" /projappl/<project>/stable-marl-rf-data.sif docker://ghcr.io/ahmed-khaled-saleh/stable-marl:robofactory-data
apptainer build --bind="$TMPDIR:/tmp" /projappl/<project>/stable-marl-cpu.sif docker://ghcr.io/ahmed-khaled-saleh/stable-marl:main
# on roihu-gpu.csc.fi: training on the GH200s (aarch64 build of :main)
apptainer build --bind="$TMPDIR:/tmp" /projappl/<project>/stable-marl-gpu.sif docker://ghcr.io/ahmed-khaled-saleh/stable-marl:main
```

RoboFactory data, one job of an array per shard (CPU partition, submitted from `roihu-cpu`):

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-9

srun apptainer exec --bind="$(csc-common-bind)" --env STABLEMARL_HOME=/scratch/<project>/stable_marl \
    /projappl/<project>/stable-marl-rf-data.sif \
    python scripts/data/collect_robofactory.py env.task=LiftBarrier expert_episodes=15 shard=$SLURM_ARRAY_TASK_ID
```

Training (GPU partition, submitted from `roihu-gpu`):

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00

srun apptainer exec --nv --bind="$(csc-common-bind)" --env STABLEMARL_HOME=/scratch/<project>/stable_marl \
    /projappl/<project>/stable-marl-gpu.sif python scripts/train/gcrl.py ...
```

Partition names and limits are CSC's (check `sinfo` / their docs); run from your checkout of the repository
(`--env PYTHONPATH=$PWD` to use its `stable_marl` instead of the image's).

## Checks

```bash
docker run --rm stable-marl:main python tests/run_all.py --quick
docker run --rm stable-marl:robofactory-data python scripts/examples/robofactory_tasks.py   # incl. the expert
docker run --rm --gpus all stable-marl:main vulkaninfo --summary                           # the GPU seen by Vulkan
```
