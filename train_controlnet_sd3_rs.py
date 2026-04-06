# train_controlnet_sd3_rs.py
"""Train SD3 ControlNet on remote sensing segmentation maps.

Reverted to original approach: segmaps encoded through VAE as 16-ch latent.
This matches SD3 ControlNet's native pos_embed_input (Conv2d 16→1536).
Early checkpointing is critical — best quality is typically at 500-1000 steps.

Usage:
    accelerate launch train_controlnet_sd3_rs.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --manifest_path ./data/hiucd_prepared/manifest.csv \
        --output_dir ./checkpoints/sd3_controlnet_rs \
        --resolution 512 \
        --train_batch_size 1 \
        --learning_rate 1e-5 \
        --max_train_steps 1000 \
        --checkpointing_steps 200 \
        --mixed_precision bf16
"""

import argparse
import logging
import os

import torch
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

from rs_data.rs_dataset import RSControlNetDataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD3 ControlNet for RS segmap conditioning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest.csv from prepare_hiucd.py")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sd3_controlnet_rs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--wandb_project", type=str, default="flowedit-rs-controlnet")
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
    scheduler = pipe.scheduler

    # Initialize ControlNet from transformer (standard zero-conv init)
    controlnet = SD3ControlNetModel.from_transformer(transformer)

    # Freeze everything except controlnet
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    controlnet.train()

    # Dataset and DataLoader
    dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="train")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Optimizer — only ControlNet params
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    # Move frozen models to device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Pre-compute text embeddings to avoid multi-GPU issues with pipe.encode_prompt()
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

    # Free text encoders from GPU
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

                # Encode segmap through VAE (same as image encoding)
                # SD3 ControlNet's pos_embed_input expects 16-ch VAE latent
                cond_pixel_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    controlnet_cond = vae.encode(cond_pixel_values).latent_dist.sample()
                    controlnet_cond = (controlnet_cond - vae.config.shift_factor) * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                t = torch.rand(bsz, device=latents.device, dtype=weight_dtype)
                timestep = t * 1000

                # Create noisy latents
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

                # Log metrics
                if args.report_to != "none":
                    accelerator.log({"train/loss": current_loss, "train/lr": current_lr}, step=global_step)

                # Checkpointing — save frequently, pick best checkpoint later
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                if global_step >= args.max_train_steps:
                    break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(args.output_dir, "controlnet_final"))
        logger.info(f"Training complete. Final model saved to {args.output_dir}/controlnet_final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
