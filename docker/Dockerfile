# stable-marl images (see docker/README.md). Build from the repository root:
#
#   docker build -f docker/Dockerfile --target main -t stable-marl:main .
#   docker build -f docker/Dockerfile --target robofactory-data -t stable-marl:robofactory-data .
#
# main              numpy 2: every environment, training (torch + JAX on CUDA), evaluation
# robofactory-data  numpy < 2: RoboFactory's motion-planning expert (mplib 0.1.1), i.e. scripts/data/collect_robofactory.py

# Debian trixie: Mesa >= 23 (lavapipe in bookworm's Mesa 22.3 is rejected by SAPIEN's renderer)
FROM python:3.11-slim-trixie AS base

# Vulkan / EGL / GL for ManiSkill's renderer (NVIDIA through the configs below, Mesa's lavapipe on CPU),
# git for ManiSkill (GitPython), ffmpeg for videos
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers vulkan-tools libegl1 libgl1 libglvnd0 libglib2.0-0t64 libxext6 libx11-6 \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ROBOFACTORY_ASSET_DIR=/opt/robofactory/assets

WORKDIR /opt/stable-marl
COPY . .


FROM base AS robofactory-data
# CPU torch is enough: RoboFactory runs on ManiSkill's CPU simulation (parallelize with shards)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[render,robofactory]" "numpy<2" \
    && python -c "import numpy, mplib; assert numpy.__version__ < '2', numpy.__version__" \
    && python -c "from stable_marl.envs.robofactory import download_assets; download_assets()"
CMD ["bash"]


FROM base AS main
# torch from PyPI bundles CUDA; JAX_CUDA picks the CUDA build of JAX for the gcrl baselines (cuda12 / cuda13)
ARG JAX_CUDA=cuda12
RUN pip install torch \
    && pip install -e ".[render,train,gcrl,vmas,robofactory]" "numpy>=2" "jax[${JAX_CUDA}]" nbdev \
    && python -c "import numpy; assert numpy.__version__ >= '2', numpy.__version__" \
    && python -c "from stable_marl.envs.robofactory import download_assets; download_assets()"
CMD ["bash"]
