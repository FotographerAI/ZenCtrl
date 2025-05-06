# Recycled from Ominicontrol and modified to accept an extra condition.
# While Zenctrl pursued a similar idea, it diverged structurally. 
# We appreciate the clarity of Omini's implementation and decided to align with it.

import torch
from typing import Union, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image


# from pipeline_tools import encode_images
from .pipeline_tools import encode_images

condition_dict = {
    "subject": 4,
    "sr": 10,
    "cot": 12,
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        position_delta=None,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        self.position_delta = position_delta

        
    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        if condition_type == "subject":
            return raw_img
        elif condition_type == "sr":
            return raw_img
        elif condition_type == "cot":
            return raw_img
        return self.condition

    
    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]
    
    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in [
            "subject",
            "sr",
            "cot"
        ]:
            if empty:
                # make the condition black
                e_condition = Image.new("RGB", self.condition.size, (0, 0, 0))
                e_condition = e_condition.convert("RGB")
                tokens, ids = encode_images(pipe, e_condition)
            else:
                tokens, ids = encode_images(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        if self.position_delta is None and self.condition_type == "subject":
            self.position_delta = [0, -self.condition.size[0] // 16]
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id
