# train_controlnet_sd3_baseline.py
"""Train SD3 ControlNet baseline for RS change generation.

Baseline approach: generate post-image from pure noise, conditioned on
(pre_image, pre_segmap, post_segmap) fused into a single ControlNet input.

Three VAE-encoded conditions (each 16-ch) are concatenated (48-ch) and
projected to 16-ch by a learnable ConditionFuser, then fed to standard
SD3 ControlNet.

Usage:
    accelerate launch train_controlnet_sd3_baseline.py \
        --hiucd_root /path/to/Hi-UCD \
        --output_dir ./checkpoints/sd3_baseline_controlnet \
        --max_train_steps 5000 --checkpointing_steps 500 --mixed_precision bf16
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    SD3ControlNetModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rs_data.rs_dataset_bitemporal import HiUCDBiTemporalDataset

logger = get_logger(__name__)


class ConditionFuser(nn.Module):
    """Fuses three VAE-encoded conditions (48-ch) into ControlNet input (16-ch).

    Input layout: [pre_img(16ch), pre_seg(16ch), post_seg(16ch)]
    Initialized so pre_img channels pass through as identity (ControlNet
    receives pre-image signal from step 0), while segmap channels start
    at zero and are learned during training.
    """

    def __init__(self, in_channels=48, out_channels=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Identity init for pre_image channels: output ≈ pre_img_latent at start
        with torch.no_grad():
            for i in range(min(out_channels, 16)):
                self.proj.weight[i, i, 0, 0] = 1.0

    def forward(self, pre_img_latent, pre_seg_latent, post_seg_latent):
        cat = torch.cat([pre_img_latent, pre_seg_latent, post_seg_latent], dim=1)
        return self.proj(cat)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD3 ControlNet baseline for RS change generation")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--hiucd_root", type=str, required=True, help="Path to Hi-UCD dataset root")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sd3_baseline_controlnet")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--wandb_project", type=str, default="flowedit-rs-baseline")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


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

    if accelerator.is_main_process and args.report_to == "wandb":
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))

    # Load base pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.float16,
    )
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # Initialize ControlNet from transformer (standard zero-conv init)
    controlnet = SD3ControlNetModel.from_transformer(transformer)

    # Condition fuser: projects concatenated 48-ch condition to 16-ch
    condition_fuser = ConditionFuser(in_channels=48, out_channels=16)

    # Freeze everything except controlnet and fuser
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.train()
    condition_fuser.train()

    # Dataset
    dataset = HiUCDBiTemporalDataset(args.hiucd_root, split=args.split, resolution=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Optimizer — ControlNet + fuser params
    trainable_params = list(controlnet.parameters()) + list(condition_fuser.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    controlnet, condition_fuser, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, condition_fuser, optimizer, dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Pre-compute text embeddings
    logger.info("Pre-computing text embeddings...")
    pipe.to(accelerator.device)
    unique_prompts = list(set(dataset.prompts))
    prompt_embed_cache = {}
    with torch.no_grad():
        for prompt in tqdm(unique_prompts, desc="Encoding prompts", disable=not accelerator.is_local_main_process):
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None,
                device=accelerator.device, do_classifier_free_guidance=False,
            )
            prompt_embed_cache[prompt] = (prompt_embeds.cpu(), pooled_prompt_embeds.cpu())

    # Free text encoders
    for comp in [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]:
        if comp is not None:
            comp.cpu()
    del pipe
    torch.cuda.empty_cache()
    logger.info(f"Cached {len(prompt_embed_cache)} unique prompt embeddings")

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training", disable=not accelerator.is_local_main_process)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(controlnet, condition_fuser):
                # VAE-encode post-image (generation target)
                post_pixels = batch["post_image"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(post_pixels).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                # VAE-encode three conditions
                pre_img_pixels = batch["pre_image"].to(dtype=weight_dtype)
                pre_seg_pixels = batch["pre_segmap"].to(dtype=weight_dtype)
                post_seg_pixels = batch["post_segmap"].to(dtype=weight_dtype)
                with torch.no_grad():
                    pre_img_lat = vae.encode(pre_img_pixels).latent_dist.sample()
                    pre_img_lat = (pre_img_lat - vae.config.shift_factor) * vae.config.scaling_factor
                    pre_seg_lat = vae.encode(pre_seg_pixels).latent_dist.sample()
                    pre_seg_lat = (pre_seg_lat - vae.config.shift_factor) * vae.config.scaling_factor
                    post_seg_lat = vae.encode(post_seg_pixels).latent_dist.sample()
                    post_seg_lat = (post_seg_lat - vae.config.shift_factor) * vae.config.scaling_factor

                # Fuse conditions: 3 x 16-ch → 16-ch
                controlnet_cond = condition_fuser(pre_img_lat, pre_seg_lat, post_seg_lat)

                # Text embeddings
                captions = batch["caption"]
                prompt_embeds = torch.cat([prompt_embed_cache[c][0] for c in captions]).to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = torch.cat([prompt_embed_cache[c][1] for c in captions]).to(accelerator.device, dtype=weight_dtype)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                t = torch.rand(bsz, device=latents.device, dtype=weight_dtype)
                timestep = t * 1000

                # Create noisy latents (flow matching forward process)
                sigmas = t.view(-1, 1, 1, 1)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # Target velocity
                target = noise - latents

                # ControlNet forward
                controlnet_block_samples = controlnet(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )[0]

                # Transformer forward with ControlNet conditioning
                controlnet_block_samples = [s.to(dtype=weight_dtype) for s in controlnet_block_samples]
                model_pred = transformer(
                    hidden_states=noisy_latents.to(dtype=weight_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    pooled_projections=pooled_prompt_embeds.to(dtype=weight_dtype),
                    block_controlnet_hidden_states=controlnet_block_samples,
                    return_dict=False,
                )[0]

                # Flow matching loss
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

                if args.report_to != "none":
                    accelerator.log({"train/loss": current_loss, "train/lr": current_lr}, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    accelerator.unwrap_model(controlnet).save_pretrained(save_dir)
                    torch.save(
                        accelerator.unwrap_model(condition_fuser).state_dict(),
                        os.path.join(save_dir, "condition_fuser.pt"),
                    )
                    logger.info(f"Saved checkpoint to {save_dir}")

                if global_step >= args.max_train_steps:
                    break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(final_dir)
        torch.save(
            accelerator.unwrap_model(condition_fuser).state_dict(),
            os.path.join(final_dir, "condition_fuser.pt"),
        )
        logger.info(f"Training complete. Final model saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
