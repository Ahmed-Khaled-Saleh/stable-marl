# Docker images

Two images, built from `docker/Dockerfile` (targets of one file, sharing the system layers):

| target | numpy | for |
|---|---|---|
| `main` | 2 | everything except RoboFactory's expert: all environments (MultiGrid, RWARE, VMAS, RoboFactory), data collection with the other experts and random policies, training (torch and JAX on CUDA: LeWM, the gcrl baselines, MAMBA), evaluation, notebooks (nbdev) |
| `robofactory-data` | < 2 | RoboFactory's motion-planning expert, i.e. `scripts/data/collect_robofactory.py`: its planner (mplib 0.1.1, pinned by ManiSkill) crashes with numpy 2, and JAX needs numpy 2, so they cannot share an environment |

Both have the repository at `/opt/stable-marl` (installed in editable mode) and RoboFactory's 3D assets
(`/opt/robofactory/assets`, so compute nodes need no internet), and both render ManiSkill scenes on NVIDIA
GPUs (Vulkan) or, without one, on the CPU (Mesa's lavapipe, slower).

## Prebuilt images (GitHub Container Registry)

The `Docker images` workflow (`.github/workflows/docker.yml`) builds both images at every push to master
that changes the code, checks them (imports, the environment examples, RoboFactory's expert on the CPU
renderer) and pushes them as `ghcr.io/ahmed-khaled-saleh/stable-marl:main` and `:robofactory-data` (plus
`:main-<commit>` / `:robofactory-data-<commit>` to pin a version). Run it by hand from the Actions tab
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

## Checks

```bash
docker run --rm stable-marl:main python tests/run_all.py --quick
docker run --rm stable-marl:robofactory-data python scripts/examples/robofactory_tasks.py   # incl. the expert
docker run --rm --gpus all stable-marl:main vulkaninfo --summary                           # the GPU seen by Vulkan
```
