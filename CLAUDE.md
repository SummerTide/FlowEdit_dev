# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlowEdit is an inversion-free text-based image editing method using pre-trained flow models (ICCV 2025 Best Student Paper). It supports two model backends: **FLUX** (black-forest-labs/FLUX.1-dev) and **Stable Diffusion 3** (stabilityai/stable-diffusion-3-medium-diffusers).

Paper: https://arxiv.org/abs/2412.08629

## Setup

```bash
pip install torch diffusers transformers accelerate sentencepiece protobuf
```

Known compatibility: `diffusers==0.30.1`, CUDA 12.4. Newer diffusers versions may break things.

## Running

```bash
# FLUX backend
python run_script.py --exp_yaml FLUX_exp.yaml

# Stable Diffusion 3 backend
python run_script.py --exp_yaml SD3_exp.yaml

# Specific GPU
python run_script.py --device_number 1 --exp_yaml FLUX_exp.yaml
```

Outputs go to `outputs/<exp_name>/<model_type>/src_<img>/tar_<idx>/`.

## Architecture

The codebase is minimal — two Python files and YAML configs:

- **`run_script.py`** — Entry point. Loads a pipeline (FLUX or SD3) from HuggingFace diffusers, reads experiment + dataset YAML configs, encodes source images through the VAE, calls the appropriate editing function, decodes and saves results.
- **`FlowEdit_utils.py`** — Core algorithm. Contains `FlowEditFLUX()` and `FlowEditSD3()` which implement the FlowEdit ODE. Key helpers: `calc_v_flux()` / `calc_v_sd3()` compute velocity predictions via the transformer, `scale_noise()` handles forward process noise scaling, `calculate_shift()` computes FLUX-specific timestep shifting.

## YAML Configuration

Two levels of YAML config:

1. **Experiment YAML** (`FLUX_exp.yaml`, `SD3_exp.yaml`) — hyperparameters: `T_steps`, `n_avg`, `src_guidance_scale`, `tar_guidance_scale`, `n_min`, `n_max`, `seed`, and a pointer to the dataset YAML.
2. **Dataset/edits YAML** (`edits.yaml`) — per-image entries with `input_img`, `source_prompt`, `target_prompts` (list), and `target_codes`.

## Key Hyperparameters

- `n_min` / `n_max` — Control which timesteps use the FlowEdit delta-velocity ODE vs. SDEDIT-style generation. Steps outside `[T_steps - n_max, T_steps - n_min]` are skipped or switch to direct sampling.
- `n_avg` — Number of noise samples averaged per step (controls quality vs. speed).
- `src_guidance_scale` / `tar_guidance_scale` — CFG scales for source and target prompts.
- `T_steps` — Total diffusion steps (28 typical for FLUX, 50 for SD3).

## Data

`Data/` contains evaluation assets: 1024x1024 images from DIV2K and royalty-free sources, with prompts in `Data/flowedit.yaml`. Academic use only.
