#!/usr/bin/env bash

# Recommended settings
export WANDB_MODE=disabled # Disable wandb entirely
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING

# Load environment variables + commands
source .env
source commands.sh

# Sync dependencies
uv sync --dev
uv pip install --no-deps -e verl/ 