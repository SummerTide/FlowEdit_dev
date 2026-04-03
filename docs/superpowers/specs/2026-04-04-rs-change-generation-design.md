# FlowEdit for Remote Sensing Change Generation — Design Spec

## 1. Problem Statement

Remote sensing change generation is traditionally modeled as conditional generation: starting from random noise, using the pre-temporal image as a control condition plus text/semantic maps to generate the post-temporal image. This approach discards the structural information in the pre-image by routing through Gaussian noise.

**Core insight**: Change generation should be modeled as a direct distribution transport from pre to post, not a noise-mediated conditional generation. FlowEdit's inversion-free ODE provides exactly this mechanism.

**Goal**: Adapt FlowEdit to generate realistic post-temporal remote sensing images from pre-temporal images, conditioned on semantic segmentation maps and text prompts, for data augmentation in change detection tasks.

## 2. Architecture

### Overview

```
SD3 (frozen) + ControlNet (trainable, semantic map condition) + FlowEdit ODE
```

Two-stage pipeline:
- **Stage 1**: Train a ControlNet on single-temporal (image, segmap, text) triplets
- **Stage 2**: Use FlowEdit's V_delta ODE with ControlNet-conditioned velocity fields for pre→post generation

### Input/Output Mapping

| Input | Role |
|---|---|
| `pre_img` | VAE encode → `x_src` → ODE initial state + per-step `zt_src` construction |
| `seg_pre` | ControlNet condition when computing `V_src` |
| `seg_post` | ControlNet condition when computing `V_tar` |
| `text_pre` | SD3 text encoder condition for `V_src` |
| `text_post` | SD3 text encoder condition for `V_tar` |

| Output | Description |
|---|---|
| `post_img_generated` | Generated post-temporal image (1024x1024) |

### Why Pre-Image Is Not an Explicit Condition

In FlowEdit, the source image participates implicitly:
1. **ODE initialization**: `zt_edit = x_src` — the editing trajectory starts from the pre image
2. **Per-step noise construction**: `zt_src = (1-t)*x_src + t*noise` — pre image structure is injected at every timestep

This is fundamentally different from conditional generation where a reference image is fed as model input. The pre image IS the trajectory, not a condition on it.

## 3. Dataset

### Primary Dataset: Hi-UCD

- Resolution: 1024x1024 (matches SD3 native resolution)
- Samples: ~1,293 bi-temporal pairs
- Labels: 9-class urban semantic segmentation + change maps
- License: Academic use

### Training Data Construction

Each bi-temporal pair (pre, post) is split into two independent single-temporal samples:
- `(pre_img, seg_pre, text_pre)`
- `(post_img, seg_post, text_post)`

Total training samples: ~2,586

### Text Prompt Generation

Template-based, derived from semantic class statistics in each segmentation map:

```
Input:  seg_map with {building: 70%, road: 20%, vegetation: 10%}
Output: "an aerial remote sensing image with buildings, roads and vegetation"
```

Classes ordered by area ratio, top-3 included in prompt.

## 4. Stage 1: ControlNet Training

### Network Structure

```
SD3 Transformer (frozen)
  ├── Normal forward: noise_latent + text_embed
  └── Receives residual features from ControlNet
       ↑
ControlNet (trainable)
  ├── Input: seg_map (class index → RGB color encoding, 3 channels)
  └── Output: per-layer residual features via zero-conv
```

### Semantic Map Encoding

Category index → fixed RGB color mapping (consistent with dataset visualization conventions). Input as 3-channel image to ControlNet, matching standard ControlNet conditioning (canny, depth, etc.).

### Training Configuration

| Item | Setting |
|---|---|
| Base model | SD3-medium (frozen) |
| Trainable params | ControlNet branch + zero-conv layers |
| Condition input | Semantic segmentation map (RGB encoded, 3ch) |
| Text condition | Template-generated prompt (co-input during training) |
| Training objective | Standard flow-matching loss (velocity prediction) |
| Data | Hi-UCD single-temporal split, ~2,586 samples |
| Resolution | 1024x1024 |
| Batch size | 1-2 (SD3 is memory-heavy) |
| Learning rate | 1e-5, warmup + cosine decay |
| Epochs | ~100 (small dataset, needs sufficient training) |
| Data augmentation | Random flip, rotation, color jitter (to mitigate overfitting) |

### Compatibility Note

The project pins `diffusers==0.30.1`. SD3 ControlNet support may require upgrading diffusers or manual adaptation. This needs verification before implementation.

## 5. Stage 2: FlowEdit Inference (Change Generation)

### Code Modifications

**Modified function**: `calc_v_sd3()` — inject ControlNet condition into velocity computation.

Before:
```python
noise_pred = pipe.transformer(
    hidden_states=latent_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
)
```

After:
```python
controlnet_output = pipe.controlnet(
    hidden_states=latent_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    controlnet_cond=seg_map_encoded,
)

noise_pred = pipe.transformer(
    hidden_states=latent_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    block_controlnet_hidden_states=controlnet_output,
)
```

**Modified function**: `FlowEditSD3()` — add `seg_pre`, `seg_post` parameters; pass corresponding segmap to `calc_v_sd3()` when computing `V_src` and `V_tar` respectively.

The core FlowEdit ODE logic (V_delta averaging, ODE stepping) remains completely unchanged.

### Inference Flow

```
1. pre_img → VAE encode → x_src
2. seg_pre  → RGB encode → ControlNet encoder → seg_pre_feat
3. seg_post → RGB encode → ControlNet encoder → seg_post_feat
4. text_pre/post → SD3 text encoder → prompt embeddings

5. FlowEdit ODE loop (T_steps=50):
   for each timestep t in [t_nmax, ..., t_0]:
     noise = randn_like(x_src)
     zt_src = (1-t)*x_src + t*noise
     zt_tar = zt_edit + zt_src - x_src
     V_src  = SD3+CtrlNet(zt_src, text_pre,  seg_pre_feat)
     V_tar  = SD3+CtrlNet(zt_tar, text_post, seg_post_feat)
     V_delta_avg += (1/n_avg) * (V_tar - V_src)
     zt_edit += dt * V_delta_avg

6. zt_edit → VAE decode → post_img_generated (1024x1024)
```

### Hyperparameters (Initial Values for RS)

| Parameter | Value | Rationale |
|---|---|---|
| `T_steps` | 50 | SD3 default |
| `n_max` | 30-40 | RS changes can be significant (farmland→building), needs strong editing |
| `n_min` | 0-5 | Conservative start, minimal pure-sampling phase |
| `n_avg` | 3-5 | RS textures are complex, more averaging for stability |
| `src_guidance_scale` | 3.5 | Default starting point |
| `tar_guidance_scale` | 10-15 | May need strong guidance to ensure semantic changes |

## 6. Evaluation

### 6.1 Quantitative Quality (Generation Realism)

| Metric | Method | Purpose |
|---|---|---|
| FID | Generated post set vs. real post set | Distribution-level realism |
| LPIPS | Per-pair: generated post vs. real post | Perceptual structure similarity |
| SSIM / PSNR | Per-pair: generated post vs. real post | Pixel-level reference metrics |

Baselines for comparison:
- **SDEdit**: Noise pre to intermediate timestep, denoise with post condition
- **Pure ControlNet generation**: From random noise with seg_post + text_post (no pre reference)
- **FlowEdit (no ControlNet)**: Text-only condition, no semantic map

### 6.2 Downstream Effectiveness (Data Augmentation)

```
Experiment design:

(a) Baseline:    real training set → train CD model → test accuracy
(b) Augmented:   real training set + generated data → train CD model → test accuracy
(c) Augmentation ratios: 25% / 50% / 100% generated data

Small-sample experiment:
(d) 10% / 25% real data + generated data → compare with full real data baseline

CD models: BIT, ChangeFormer, or TinyCD
Metrics:   F1, IoU, OA (standard CD metrics)
```

### 6.3 Qualitative Controllability Analysis

Visualize and analyze:
- **Semantic consistency**: Do generated regions match seg_post class assignments?
- **Unchanged region preservation**: Are textures consistent where seg_pre == seg_post?
- **Change boundary quality**: Are transitions natural and smooth?
- **Diversity**: Different random seeds for same (pre, seg_post) → reasonable variation?

## 7. Implementation Roadmap

1. **Data preparation**: Hi-UCD download, single-temporal split, text prompt generation
2. **ControlNet training**: SD3 ControlNet on single-temporal RS data
3. **FlowEdit adaptation**: Modify `calc_v_sd3()` and `FlowEditSD3()` to accept ControlNet conditions
4. **Hyperparameter tuning**: Grid search on n_max, n_avg, guidance scales with small validation set
5. **Evaluation**: Quality metrics → downstream CD augmentation → qualitative analysis

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| SD3 domain gap (natural → RS) | ControlNet training bridges the gap; if insufficient, consider LoRA fine-tuning SD3 on RS data |
| Small training set (2,586 samples) | Aggressive data augmentation; monitor overfitting; consider adding LoveDA/iSAID data later |
| diffusers version compatibility | Verify SD3 ControlNet support early; may need version upgrade or manual implementation |
| Hyperparameter sensitivity | Start with paper defaults, systematic sweep on n_max and guidance scales |
| Generation diversity too low | Increase n_avg noise variance, adjust guidance scales, vary random seeds |
