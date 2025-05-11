# Recycled from Ominicontrol 

import gradio as gr
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel

from flux.condition import Condition
from flux.generate import generate
from flux.lora_controller import set_lora_scale

pipe = None
use_int8 = False
model_config = { "union_cond_attn": True, "add_cond_attn": False, "latent_lora": False, "independent_condition": False}

def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def init_pipeline():
    global pipe
    if use_int8 or get_gpu_memory() < 33:
        transformer_model = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/flux.1-schell-int8wo-improved",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
    pipe = pipe.to("cuda")
    
    # Optional: Load additional LoRA weights, put the loaded weigths here!
    pipe.load_lora_weights("weights/zen2con_1440_17000/pytorch_lora_weights.safetensors",
        adapter_name="subject")
    pipe.set_adapters(["subject"])
    
def paste_on_white_background(image: Image.Image) -> Image.Image:
    """
    Pastes a transparent image onto a white background of the same size.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Create white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white_bg.paste(image, (0, 0), mask=image)
    return white_bg.convert("RGB")  # Convert back to RGB if you don't need alpha


def process_image_and_text(image, text, steps=8, strength_sub=1.0, strength_spat=1.0, size=1024):
    # center crop image
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(
        (
            (w - min_size) // 2,
            (h - min_size) // 2,
            (w + min_size) // 2,
            (h + min_size) // 2,
        )
    )
    image = image.resize((size, size))
    image = paste_on_white_background(image) #Optional, you can remove this line if you want just make sure the size it matched.
    condition0 = Condition("subject", image, position_delta=(0, size // 16))
    condition1 = Condition("subject", image, position_delta=(0, -size // 16))
    
    if pipe is None:
        init_pipeline()
    
    with set_lora_scale(["subject"], scale=3.0):
        result_img = generate(
            pipe,
            prompt=text.strip(),
            conditions=[condition0, condition1],
            num_inference_steps=steps,
            height=1024,
            width=1024,
            condition_scale = [strength_sub,strength_spat],
            model_config=model_config,
            default_lora=True,
        ).images[0]

    return [condition0.condition, condition1.condition, result_img]


def get_samples():
    sample_list = [
        {
            "image": "samples/1.png",   #place your image path here
            "text": "A man sitting in a yellow chair drinking a cup of coffee",
        }
    ]
    return [[Image.open(sample["image"]), sample["text"]] for sample in sample_list]


demo = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2),
        gr.Slider(minimum=2, maximum=28, value=2, label="steps"),
        gr.Slider(minimum=0, maximum=2.0, value=1.0, label="strength_sub"),
        gr.Slider(minimum=0, maximum=2.0, value=1.0, label="strength_spat"),
        gr.Slider(minimum=512, maximum=2048, value=1024, label="size"),
    ],
    outputs=gr.Gallery(
                label="Outputs", show_label=False, elem_id="gallery",
                columns=[3], rows=[1], object_fit="contain", height="auto"
            ),
    title="ZenCtrl / Subject driven generation",
    examples=get_samples(),
)

if __name__ == "__main__":
    init_pipeline()
    demo.launch(
        debug=True,
        # share=True
    )
