# train_controlnet_sd3_rs.py
"""Train SD3 ControlNet on remote sensing segmentation maps.

Usage:
    accelerate launch train_controlnet_sd3_rs.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --manifest_path ./data/hiucd_prepared/manifest.csv \
        --output_dir ./checkpoints/sd3_controlnet_rs \
        --resolution 1024 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --max_train_steps 15000 \
        --validation_steps 500 \
        --checkpointing_steps 1000
"""

import argparse
import logging
import math
import os

import torch
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rs_data.rs_dataset import RSControlNetDataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD3 ControlNet for RS segmap conditioning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest.csv from prepare_hiucd.py")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sd3_controlnet_rs")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--wandb_project", type=str, default="flowedit-rs-controlnet")
    return parser.parse_args()


def save_controlnet_with_rgb_config(controlnet_model, save_path, resolution):
    """Save ControlNet and patch config.json for 3-ch RGB pos_embed_input."""
    controlnet_model.save_pretrained(save_path)
    # Update config to reflect modified pos_embed_input (3-ch RGB, patch_size=16)
    import json
    config_path = os.path.join(save_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["_rgb_controlnet"] = True
    config["_rgb_patch_size"] = controlnet_model.pos_embed_input.proj.kernel_size[0]
    config["_rgb_in_channels"] = controlnet_model.pos_embed_input.proj.in_channels
    config["_rgb_resolution"] = resolution
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_rgb_controlnet(checkpoint_path, dtype=torch.float16, resolution=512):
    """Load ControlNet with 3-ch RGB pos_embed_input from checkpoint.

    Strategy: detect actual weight shapes from the safetensors file,
    build model with matching pos_embed_input, then load all weights.
    """
    import json
    import safetensors.torch
    from diffusers.models.embeddings import PatchEmbed

    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Load state dict to inspect actual weight shapes
    ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")]
    state_dict = safetensors.torch.load_file(os.path.join(checkpoint_path, ckpt_files[0]))

    # Detect pos_embed_input shape from weights
    pe_weight = state_dict["pos_embed_input.proj.weight"]
    actual_out_ch, actual_in_ch, actual_k, _ = pe_weight.shape  # e.g. [1536, 3, 16, 16]

    # Build model with default config (16-ch, patch=2)
    controlnet = SD3ControlNetModel(**{k: v for k, v in config.items() if not k.startswith("_")})

    # Replace pos_embed_input to match saved weights
    if actual_in_ch != controlnet.pos_embed_input.proj.in_channels or actual_k != controlnet.pos_embed_input.proj.kernel_size[0]:
        controlnet.pos_embed_input = PatchEmbed(
            height=resolution,
            width=resolution,
            patch_size=actual_k,
            in_channels=actual_in_ch,
            embed_dim=actual_out_ch,
            pos_embed_type="sincos",
            pos_embed_max_size=config.get("pos_embed_max_size"),
        )

    # Load all weights — now shapes match (strict=False for pos_embed which may not be saved)
    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if missing:
        # Only pos_embed_input.pos_embed is expected to be missing (sincos, not learned)
        for k in missing:
            if "pos_embed" not in k:
                raise RuntimeError(f"Unexpected missing key: {k}")
    controlnet = controlnet.to(dtype=dtype)
    return controlnet


def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        log_with=args.report_to if args.report_to != "none" else None,
        kwargs_handlers=[ddp_kwargs],
    )
    logging.basicConfig(level=logging.INFO)
    logger.info(f"[RANK {accelerator.process_index}] DDP find_unused_parameters=True, static_graph=True")

    if accelerator.is_main_process and args.report_to == "wandb":
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
        )

    # Load base pipeline components
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_3 = pipe.text_encoder_3
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3
    scheduler = pipe.scheduler

    # Initialize ControlNet from transformer
    controlnet = SD3ControlNetModel.from_transformer(transformer)

    # Initialize controlnet_blocks (zero-conv) with small nonzero values
    # instead of zeros. With zero init, gradients cannot flow back to
    # ControlNet's internal transformer blocks (grad * 0 = 0).
    # Small init allows gradients to flow from the start.
    import torch.nn as nn
    for block in controlnet.controlnet_blocks:
        if isinstance(block, nn.Linear):
            nn.init.normal_(block.weight, std=1e-5)
            if block.bias is not None:
                nn.init.zeros_(block.bias)

    # Replace pos_embed_input to accept 3-channel RGB instead of 16-channel VAE latent
    # This avoids encoding semantic segmaps through VAE, which destroys their structure
    #
    # Dimension math for 512x512 RGB input:
    #   hidden_states = 64x64 latent / patch_size=2 = 32x32 = 1024 tokens
    #   RGB input needs same 1024 tokens: 512 / patch_size = 32 -> patch_size = 16
    from diffusers.models.embeddings import PatchEmbed
    old_pe = controlnet.pos_embed_input
    vae_scale_factor = pipe.vae_scale_factor  # 8
    rgb_patch_size = old_pe.proj.kernel_size[0] * vae_scale_factor  # 2 * 8 = 16
    controlnet.pos_embed_input = PatchEmbed(
        height=args.resolution,
        width=args.resolution,
        patch_size=rgb_patch_size,
        in_channels=3,  # RGB input instead of 16-ch latent
        embed_dim=old_pe.proj.out_channels,
        pos_embed_type="sincos",
        pos_embed_max_size=old_pe.pos_embed_max_size,
    )

    # Freeze base models
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)

    # Add LoRA to transformer for RS domain adaptation
    # This lets the transformer learn RS-specific features while keeping most weights frozen
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    transformer.train()

    controlnet.train()

    # Dataset and DataLoader
    dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="train")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Optimizer — train both ControlNet and transformer LoRA params
    trainable_params = list(controlnet.parameters()) + list(filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator — include transformer (has LoRA trainable params)
    controlnet, transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, transformer, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Pre-compute text embeddings for all unique prompts to avoid per-step encoding
    # This also avoids multi-GPU issues with pipe.encode_prompt()
    logger.info("Pre-computing text embeddings...")
    pipe.to(accelerator.device)
    unique_prompts = list(set(s["text_prompt"] for s in dataset.samples))
    prompt_embed_cache = {}
    with torch.no_grad():
        for prompt in tqdm(unique_prompts, desc="Encoding prompts", disable=not accelerator.is_local_main_process):
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None,
                device=accelerator.device, do_classifier_free_guidance=False,
            )
            prompt_embed_cache[prompt] = (prompt_embeds.cpu(), pooled_prompt_embeds.cpu())

    # Free text encoders from GPU — no longer needed
    if text_encoder is not None:
        text_encoder.cpu()
    if text_encoder_2 is not None:
        text_encoder_2.cpu()
    if text_encoder_3 is not None:
        text_encoder_3.cpu()
    del pipe
    torch.cuda.empty_cache()
    logger.info(f"Cached {len(prompt_embed_cache)} unique prompt embeddings")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training", disable=not accelerator.is_local_main_process)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(controlnet):
                # Encode images to latent space
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                # Look up pre-computed text embeddings
                captions = batch["caption"]
                prompt_embeds = torch.cat([prompt_embed_cache[c][0] for c in captions]).to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = torch.cat([prompt_embed_cache[c][1] for c in captions]).to(accelerator.device, dtype=weight_dtype)

                # Prepare conditioning image — feed RGB segmap directly (3 channels)
                # pos_embed_input has been replaced to accept 3-ch RGB instead of 16-ch VAE latent
                controlnet_cond = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample random timestep for each sample (flow matching uses uniform [0, 1])
                t = torch.rand(bsz, device=latents.device, dtype=weight_dtype)
                timestep = t * 1000  # SD3 uses timestep in [0, 1000]

                # Create noisy latents (linear interpolation for flow matching)
                sigmas = t.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # Target velocity
                target = noise - latents

                # ControlNet + Transformer forward under autocast
                # IMPORTANT: do NOT manually cast tensors — autocast preserves the
                # computation graph so gradients flow back to all ControlNet params
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    controlnet_block_samples = controlnet(
                        hidden_states=noisy_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                    )[0]

                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=controlnet_block_samples,
                        return_dict=False,
                    )[0]

                # Flow matching loss (in fp32 for stability)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                current_loss = loss.detach().item()
                current_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(loss=current_loss, lr=current_lr)

                # Log metrics
                if args.report_to != "none":
                    accelerator.log({"train/loss": current_loss, "train/lr": current_lr}, step=global_step)

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_controlnet_with_rgb_config(accelerator.unwrap_model(controlnet), save_path, args.resolution)
                    logger.info(f"Saved checkpoint to {save_path}")

                # Validation disabled during training — run separately after training
                # (40GB GPU cannot hold training state + full validation pipeline simultaneously)

                if global_step >= args.max_train_steps:
                    break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_unwrapped = accelerator.unwrap_model(controlnet)
        save_controlnet_with_rgb_config(controlnet_unwrapped, os.path.join(args.output_dir, "controlnet_final"), args.resolution)
        logger.info(f"Training complete. Final model saved to {args.output_dir}/controlnet_final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
