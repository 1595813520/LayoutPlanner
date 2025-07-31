# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext
from PIL import Image

import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    ViTImageProcessor,
    ViTMAEModel,
    LlamaTokenizer
)

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.models.mllm.seed_x import ContinuousLVLM


class DiffSenseiPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        scheduler: KarrasDiffusionSchedulers,
        # manga modified model
        unet: UNetMangaModel,
        image_encoder: CLIPVisionModelWithProjection,
        # not used others
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()

    def register_manga_modules(
        self,
        magi_image_encoder: ViTMAEModel,
        image_proj_model: Resampler,
    ):
        self.magi_image_encoder = magi_image_encoder
        self.image_proj_model = image_proj_model

    def check_inputs(
        self,
        prompt,
        prompt_2,
        ip_images,
        ip_image_embeds,
        ip_bbox
    ):
        if prompt is None:
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")
        elif prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"`prompt` has to be of type `str` but is {type(prompt)}")
        elif prompt_2 is not None and not isinstance(prompt_2, str):
            raise ValueError(f"`prompt_2` has to be of type `str` but is {type(prompt_2)}")

        if len(ip_images) > 0 and ip_image_embeds is not None:
            raise ValueError(f"`ip_images` and `ip_image_embeds` can not be input together!")

        num_ips = len(ip_image_embeds) if ip_image_embeds is not None else len(ip_images)
        
        if num_ips != len(ip_bbox):
            raise ValueError(f"`ip_images` must have the same length as `ip_bbox`. But they are in length {num_ips} and {len(ip_bbox)}!")
        
    def prepare_ip_image_embeds(
        self,
        ip_images,
        ip_image_embeds,
        ip_bbox,
        num_samples,
    ):  
        max_num_ips = self.unet.config.max_num_ips
        ip_images = ip_images[:max_num_ips]
        if ip_image_embeds is not None:
            ip_image_embeds = ip_image_embeds[:max_num_ips]
        ip_bbox = ip_bbox[:max_num_ips]
        num_ips = len(ip_images)

        # pad ip_images and ip_bbox
        while len(ip_images) < max_num_ips:
            ip_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
        while len(ip_bbox) < max_num_ips:
            ip_bbox.append([0.0, 0.0, 0.0, 0.0])
        
        # encode ip_images
        clip_ip_images = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
        magi_ip_images = self.magi_image_processor(images=ip_images, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_ip_images.to(self._execution_device, dtype=self.image_encoder.dtype), output_hidden_states=True).hidden_states[-2].unsqueeze(0)
        magi_image_embeds = self.magi_image_encoder(magi_ip_images.to(self._execution_device, dtype=self.magi_image_encoder.dtype)).last_hidden_state[:, 0].unsqueeze(0)

        # if number of ip_images is less than max_num_ips, pad the image_embeds with all zero tensors
        clip_image_embeds[0][num_ips:] = torch.zeros_like(clip_image_embeds[0][num_ips:])
        magi_image_embeds[0][num_ips:] = torch.zeros_like(magi_image_embeds[0][num_ips:])
        image_embeds = self.image_proj_model(clip_image_embeds.to(dtype=self.image_proj_model.dtype()), magi_image_embeds.to(dtype=self.image_proj_model.dtype()) if magi_image_embeds is not None else None) 

        negative_image_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds.to(dtype=self.image_proj_model.dtype())), torch.zeros_like(magi_image_embeds).to(dtype=self.image_proj_model.dtype()))

        ip_bbox = torch.Tensor(ip_bbox).unsqueeze(0).to(self._execution_device)
        negative_ip_bbox = torch.zeros_like(ip_bbox)
        
        image_embeds = image_embeds.view(1, self.unet.config.num_vision_tokens + max_num_ips * self.unet.config.num_vision_tokens, image_embeds.shape[-1]) # [1, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]
        
        # Paste generated ip image embeds in to image_embeds
        if ip_image_embeds is not None:
            num_ip_image_embeds, _, dim = ip_image_embeds.shape
            image_embeds[0, self.unet.config.num_vision_tokens:(1 + num_ip_image_embeds) * self.unet.config.num_vision_tokens, :] = ip_image_embeds.view(1, -1, dim)
        
        negative_image_embeds = negative_image_embeds.view(1, self.unet.config.num_vision_tokens + max_num_ips * self.unet.config.num_vision_tokens, image_embeds.shape[-1])

        image_embeds = image_embeds.repeat(num_samples, 1, 1).to(dtype=self.unet.dtype)
        negative_image_embeds = negative_image_embeds.repeat(num_samples, 1, 1).to(dtype=self.unet.dtype)
        ip_bbox = ip_bbox.repeat(num_samples, 1, 1)
        negative_ip_bbox = negative_ip_bbox.repeat(num_samples, 1, 1)

        return negative_image_embeds, image_embeds, negative_ip_bbox, ip_bbox
    
    def prepare_dialog_bbox(
        self,
        dialog_bbox,
        num_samples,
    ):
        max_num_dialogs = self.unet.config.max_num_dialogs
        dialog_bbox = dialog_bbox[:max_num_dialogs]
        while len(dialog_bbox) < max_num_dialogs:
            dialog_bbox.append([0.0, 0.0, 0.0, 0.0])

        dialog_bbox = torch.Tensor(dialog_bbox).unsqueeze(0).to(device=self._execution_device, dtype=self.unet.dtype)
        dialog_bbox = dialog_bbox.repeat(num_samples, 1, 1)
        negative_dialog_bbox = torch.zeros_like(dialog_bbox)

        return negative_dialog_bbox, dialog_bbox

    def set_ip_scale(
        self,
        scale,
    ):
        for attn_processor in self.unet.attn_processors.values():
            if hasattr(attn_processor, "scale"):
                attn_processor.scale = scale

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        prompt_2: str = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_samples: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        min_size_step: Optional[int] = 8,
        # manga conditions
        ip_images: Optional[PipelineImageInput] = [],
        ip_image_embeds: Optional[torch.Tensor] = None,
        ip_bbox: Optional[List[List[float]]] = [],
        ip_scale: Optional[int] = 1.0,
        dialog_bbox: Optional[List[List[float]]] = [],
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if height % min_size_step != 0 or width % min_size_step != 0:
            height = (height / min_size_step) * min_size_step
            width = (width / min_size_step) * min_size_step

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            ip_images,
            ip_image_embeds,
            ip_bbox,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define hyper-parameters
        device = self._execution_device
        self.set_ip_scale(ip_scale)

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            num_samples,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator
        )

        # 6. Prepare IP image embeddings
        negative_image_embeds, image_embeds, negative_ip_bbox, ip_bbox = self.prepare_ip_image_embeds(
            ip_images,
            ip_image_embeds,
            ip_bbox,
            num_samples,
        )
        cross_attention_kwargs = {
            "bbox": torch.cat([negative_ip_bbox, ip_bbox], dim=0),
            "aspect_ratio": latents.shape[-2] / latents.shape[-1]
        }

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        negative_dialog_bbox, dialog_bbox = self.prepare_dialog_bbox(
            dialog_bbox,
            num_samples,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
            dialog_bbox = torch.cat([negative_dialog_bbox, dialog_bbox], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(num_samples, 1)
        prompt_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)
        dialog_bbox = dialog_bbox.to(device)

        # 8. Denoising loop
        self._num_timesteps = len(timesteps)
        progress_bar_context = nullcontext() if self.progress_bar_config["disable"] else self.progress_bar(total=num_inference_steps)
        with progress_bar_context:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    dialog_bbox=dialog_bbox,
                ).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image = self.image_processor.postprocess(image, output_type="pil")

        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionXLPipelineOutput(images=image)
