# Recycled from Ominicontrol and modified to accept an extra condition.
# While Zenctrl pursued a similar idea, it diverged structurally. 
# We appreciate the clarity of Omini's implementation and decided to align with it.

import torch
from typing import  Optional, Dict, Any
from .block import block_forward, single_block_forward
from .lora_controller import enable_lora
from accelerate.utils import is_torch_version
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
    logger,
)
import numpy as np


def prepare_params(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    **kwargs: dict,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    )


def tranformer_forward(
    transformer: FluxTransformer2DModel,
    condition_latents: torch.Tensor,
    extra_condition_latents: torch.Tensor,
    condition_ids: torch.Tensor,
    condition_type_ids: torch.Tensor,
    extra_condition_ids: torch.Tensor,
    extra_condition_type_ids: torch.Tensor,
    model_config: Optional[Dict[str, Any]] = {},
    c_t=0,
    **params: dict,
):
    self = transformer
    use_condition = condition_latents is not None
    use_extra_condition = extra_condition_latents is not None

    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        return_dict,
    ) = prepare_params(**params)

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    with enable_lora((self.x_embedder,), model_config.get("latent_lora", False)):
        hidden_states = self.x_embedder(hidden_states)
    condition_latents = self.x_embedder(condition_latents) if use_condition else None
    extra_condition_latents = self.x_embedder(extra_condition_latents) if use_extra_condition else None

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    cond_temb = (
        self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
        if guidance is None
        else self.time_text_embed(
            torch.ones_like(timestep) * c_t * 1000, torch.ones_like(guidance) * 1000, pooled_projections
        )
    )
    extra_cond_temb = (
        self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
        if guidance is None
        else self.time_text_embed(
            torch.ones_like(timestep) * c_t * 1000, torch.ones_like(guidance) * 1000, pooled_projections
        )
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    if use_condition:
        # condition_ids[:, :1] = condition_type_ids
        cond_rotary_emb = self.pos_embed(condition_ids)

    if use_extra_condition:
        extra_cond_rotary_emb = self.pos_embed(extra_condition_ids)


    # hidden_states = torch.cat([hidden_states, condition_latents], dim=1)

    #print("here!")
    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states, condition_latents, extra_condition_latents = (
                torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    condition_latents=condition_latents if use_condition else None,
                    extra_condition_latents=extra_condition_latents if use_extra_condition else None,
                    temb=temb,
                    cond_temb=cond_temb if use_condition else None,
                    cond_rotary_emb=cond_rotary_emb if use_condition else None,
                    extra_cond_temb=extra_cond_temb if use_extra_condition else None,
                    extra_cond_rotary_emb=extra_cond_rotary_emb if use_extra_condition else None,
                    image_rotary_emb=image_rotary_emb,
                    **ckpt_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states, condition_latents, extra_condition_latents = block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                condition_latents=condition_latents if use_condition else None,
                extra_condition_latents=extra_condition_latents if use_extra_condition else None,
                temb=temb,
                cond_temb=cond_temb if use_condition else None,
                cond_rotary_emb=cond_rotary_emb if use_condition else None,
                extra_cond_temb=cond_temb if use_extra_condition else None,
                extra_cond_rotary_emb=extra_cond_rotary_emb if use_extra_condition else None,
                image_rotary_emb=image_rotary_emb,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_block_samples[index_block // interval_control]
            )
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            result = torch.utils.checkpoint.checkpoint(
                single_block_forward,
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                **(
                    {
                        "condition_latents": condition_latents,
                        "extra_condition_latents": extra_condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "extra_cond_temb": extra_cond_temb,
                        "extra_cond_rotary_emb": extra_cond_rotary_emb,
                    }
                    if use_condition
                    else {}
                ),
                **ckpt_kwargs,
            )

        else:
            result = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                **(
                    {
                        "condition_latents": condition_latents,
                        "extra_condition_latents": extra_condition_latents,
                        "cond_temb": cond_temb,
                        "cond_rotary_emb": cond_rotary_emb,
                        "extra_cond_temb": extra_cond_temb,
                        "extra_cond_rotary_emb": extra_cond_rotary_emb,
                    }
                    if use_condition
                    else {}
                ),
            )
        if use_condition:
            hidden_states, condition_latents, extra_condition_latents = result
        else:
            hidden_states = result

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
