# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor
from diffusers import StableDiffusionUpscalePipeline
import inspect
import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType
from diffusers.models.attention_processor import AttnProcessor2_0, LoRAXFormersAttnProcessor, XFormersAttnProcessor

CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
SUP_RES = "stabilityai/stable-diffusion-x4-upscaler"

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=True) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionUpscalePipeline.from_pretrained(SUP_RES, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler.set_timesteps(75)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        #pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.low_res_scheduler = pipe.low_res_scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()
        

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.to(dtype = torch.float32)
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.auto_encoder = pipe.vae
        self.vae = pipe.vae
        CONSOLE.print("InstructPix2Pix loaded!")

    def edit_image(
        self,
        
        prompt: str,
        image: TensorType["BS", 3, "H", "W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        noise_level: float = 20,
        batch_multiplier:int = 1,
        eta:float = 0.0
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = None
        num_images_per_prompt = 1
        negative_prompt = None
        negative_prompt_embeds= None
            
        prompt_embeds = self._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        
        self.scheduler.set_timesteps(75, device=self.device)
        timesteps = self.scheduler.timesteps

        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = randn_tensor(image.shape, generator=None, device=self.device, dtype=prompt_embeds.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = torch.cat([image] * batch_multiplier * num_images_per_prompt)
        noise_level = torch.cat([noise_level] * image.shape[0])
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        generator = None
        batch_size = 1
        latents = None
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        num_channels_image = image.shape[1]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        #with torch.no_grad():
            # prepare image and image_cond latents
        #     image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        with torch.no_grad():
            with torch.autocast("cuda"):
                for i, t in enumerate(self.scheduler.timesteps):

            # predict the noise residual with unet, NO grad!
                # pred noise
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat([latent_model_input, image], dim=1)

                # predict the noise residual
                    noise_pred = self.unet(
                    latent_model_input.float(),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs= None,
                    class_labels=noise_level,
                    return_dict=False,
                    )[0]

                # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                print(latents.dtype,latent_model_input.dtype,prompt_embeds.dtype)
            
        # decode latents to get edited image
        decoded_img = self.latents_to_img(latents)
        
        return decoded_img

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = (1 / CONST_SCALE) * latents
        
        print(latents.dtype)
                
        with torch.no_grad():
            use_torch_2_0_or_xformers = self.vae.decoder.mid_block.attentions[0].processor in [
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
            ]
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
            if not use_torch_2_0_or_xformers:
                self.vae.post_quant_conv.to(latents.dtype)
                self.vae.decoder.conv_in.to(latents.dtype)
                self.vae.decoder.mid_block.to(latents.dtype)
            else:
                latents = latents.float()
            
            imgs = (self.vae.decode(latents, return_dict=False)[0]).to(dtype = torch.float32)
            print(imgs)
        imgs_to_latent = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0
        
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
