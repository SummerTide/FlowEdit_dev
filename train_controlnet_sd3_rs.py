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
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_validation_images", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )
    logging.basicConfig(level=logging.INFO)

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

    # Freeze everything except controlnet
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)

    controlnet.train()

    # Dataset and DataLoader
    dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="train")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Optimizer
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
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_2 is not None:
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_3 is not None:
        text_encoder_3.to(accelerator.device, dtype=weight_dtype)

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

                # Encode text
                captions = batch["caption"]
                with torch.no_grad():
                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                        pipe.encode_prompt(
                            prompt=captions,
                            prompt_2=None,
                            prompt_3=None,
                            device=accelerator.device,
                            do_classifier_free_guidance=False,
                        )
                    )

                # Prepare conditioning image
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
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
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
                progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(controlnet).save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                # Validation
                if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                    logger.info(f"Running validation at step {global_step}...")
                    val_dataset = RSControlNetDataset(args.manifest_path, resolution=args.resolution, split="test")
                    if len(val_dataset) > 0:
                        val_pipe = StableDiffusion3ControlNetPipeline(
                            transformer=transformer,
                            controlnet=accelerator.unwrap_model(controlnet),
                            scheduler=scheduler,
                            vae=vae,
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            text_encoder_3=text_encoder_3,
                            tokenizer=tokenizer,
                            tokenizer_2=tokenizer_2,
                            tokenizer_3=tokenizer_3,
                        )
                        val_dir = os.path.join(args.output_dir, f"validation-{global_step}")
                        os.makedirs(val_dir, exist_ok=True)
                        for vi in range(min(args.num_validation_images, len(val_dataset))):
                            sample = val_dataset[vi]
                            seg_img = Image.fromarray((sample["conditioning_pixel_values"].permute(1, 2, 0).numpy() * 255).astype("uint8"))
                            with torch.autocast("cuda"):
                                out = val_pipe(
                                    prompt=sample["caption"],
                                    control_image=seg_img,
                                    num_inference_steps=50,
                                ).images[0]
                            out.save(os.path.join(val_dir, f"val_{vi}.png"))
                            seg_img.save(os.path.join(val_dir, f"val_{vi}_seg.png"))
                        del val_pipe
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_unwrapped = accelerator.unwrap_model(controlnet)
        controlnet_unwrapped.save_pretrained(os.path.join(args.output_dir, "controlnet_final"))
        logger.info(f"Training complete. Final model saved to {args.output_dir}/controlnet_final")

    accelerator.end_training()


if __name__ == "__main__":
    main()
