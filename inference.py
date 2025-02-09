import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionXLPipeline
from controlnet_union import ControlNetModel_Union
from pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline

from controlnet_aux import (
    OpenposeDetector, 
    MidasDetector, 
    ZoeDetector, 
    LineartDetector, 
    HEDdetector, 
    MLSDdetector, 
    NormalBaeDetector, 
    SamDetector, 
    PidiNetDetector,  
)

import torch
import time

from diffusers import DiffusionPipeline

import torch
from diffusers import DiffusionPipeline
from PIL import Image
import base64
from io import BytesIO
from omegaconf import OmegaConf

from reactor_components.perform_faceswap import FaceSwapScript
from reactor_components.utils import download_ckpts

# from utils import encode_image_to_base64, decode_base64_to_image

class Processors: 
    def __init__(self, device="cuda", cache_dir="./weights"): 
        self.processors = self._init_processors(device, cache_dir)

    def nms(self, x, t, s):
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)
    
        f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    
        y = np.zeros_like(x)
    
        for f in [f1, f2, f3, f4]:
            np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    
        z = np.zeros_like(y, dtype=np.uint8)
        z[y > t] = 255
        return z
    
    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y
        
    def processor_scribble(self, source_img:np.ndarray, processor_hed, output_type='cv2',): 
        controlnet_img = processor_hed(source_img, scribble=False)
        controlnet_img.save("a hed detect path for an image")
    
        controlnet_img = np.array(controlnet_img)
        controlnet_img = nms(controlnet_img, 127, 3)
        controlnet_img = cv2.GaussianBlur(controlnet_img, (0, 0), 3)
    
        random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
        controlnet_img[controlnet_img > random_val] = 255
        controlnet_img[controlnet_img < 255] = 0
        controlnet_img = Image.fromarray(controlnet_img) if output_type == 'pil' else controlnet_img
        return controlnet_img

    def processor_softedge(self, source_img, processor_hed): 
        controlnet_img = processor_hed(source_img, scribble=False)
    
        controlnet_img = np.array(controlnet_img)
        controlnet_img = nms(controlnet_img, 127, 3)
        controlnet_img = cv2.GaussianBlur(controlnet_img, (0, 0), 3)
    
        # higher threshold, thiner line
        random_val = int(round(random.uniform(0.01, 0.10), 2) * 255)
        controlnet_img[controlnet_img > random_val] = 255
        controlnet_img[controlnet_img < 255] = 0
        controlnet_img = Image.fromarray(controlnet_img)
        return controlnet_img
        
    def processor_canny(self, source_img:np.ndarray, output_type='pil'): 
        controlnet_img = cv2.Canny(source_img, 100, 200)
        controlnet_img = self.HWC3(controlnet_img)    
        controlnet_img = Image.fromarray(controlnet_img) if output_type == 'pil' else controlnet_img
        return controlnet_img
    
    def _init_processors(self, device, cache_dir): 
        # 0 -- openpose
        # 1 -- depth
        # 2 -- hed/pidi/scribble/ted
        # 3 -- canny/lineart/anime_lineart/mlsd
        # 4 -- normal
        # 5 -- segment

        openpose_processors = {
            "openpose": OpenposeDetector.from_pretrained('lllyasviel/ControlNet', cache_dir=cache_dir),
        }
        depth_processors = {
            'zoe': ZoeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir),
            'midas': MidasDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir),
        }
        sketch_processors = {
            'pidi': PidiNetDetector.from_pretrained('lllyasviel/Annotators', cache_dir=cache_dir).to(device), 
            'hed': HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir=cache_dir).to(device), 
        }
        canny_processors = {
            'canny': self.processor_canny, 
        }
        normal_processors = {
            "normal": NormalBaeDetector.from_pretrained('lllyasviel/Annotators', cache_dir=cache_dir).to(device),
        }
        segment_processors = {
            "sam": SamDetector.from_pretrained(
                "ybelkada/segment-anything", 
                subfolder="checkpoints",
                cache_dir=cache_dir,
            )
        }
        return {
            "openpose": openpose_processors,  
            "depth": depth_processors, 
            "hed": sketch_processors, 
            "edges": canny_processors, 
            "normal": normal_processors, 
            "segment": segment_processors, 
        }

    
    def __call__(self, image:np.ndarray, processor: str = 'canny'): 
        conditions = []
        selected_processor = random.choice(
            list(self.processors[processor].values())
        )
        condition_image = selected_processor(image, output_type='pil')
        return condition_image
        
device=torch.device('cuda:0')
cache_dir = "./weights"
    
ratios = {
    "1:1": (1024, 1024)
}

class FaceSwapper:

    def __init__(self, config_path: str) -> None:
        self.cfg = OmegaConf.load(config_path)
        download_ckpts(self.cfg)
        self.script = FaceSwapScript()

    def do_swap(self, input_image: Image.Image, source_image: Image.Image) -> Image.Image:
        swapped = self.script.process(source_img=source_image, input_img=input_image, cfg=self.cfg)
        return swapped

def inference_reactor(pipe, source_image, face_image):
    out_image = pipe.do_swap(source_image, face_image)
    return out_image
    
def inference_sdxl(pipe, data, processor_pipes):
    try:
        print(f"inference: {id(pipe)}")
        # Extract required parameters from the input data
        sampling_params = data.get("sampling_parameters", {})
        adapters = data.get("adapters", {})
        
        # Setup prompt and negative prompt
        prompt = sampling_params.get("prompt", "")
        neg_prompt = sampling_params.get("neg_prompt", "")
        
        # Define image generation settings
        num_steps = sampling_params.get("num_steps", 30)
        guidance_scale = sampling_params.get("guidance_scale", 7.5)
        seed = sampling_params.get("seed", None)
        num_images_per_prompt = sampling_params.get("num_images_per_prompt", 1)
        aspect_ratio = sampling_params.get("aspect_ratio", "1:1")
        width, height = ratios[aspect_ratio]
        
        # Generate a random or fixed seed
        start_time = time.time()
        generator = torch.Generator('cuda').manual_seed(seed) if seed else torch.Generator('cuda')
        
        # Load LoRAs from local paths if specified in the adapters section
        # Set adapter weights
        start_time = time.time()
        loras = adapters.get("loras", [])
        if loras:
            lora_names = []
            lora_weights = []
            for lora in loras:
                lora_path = lora.get("url")  
                lora_weight = lora.get("weight", 1.0)
                lora_dict = lora.get("state_dict")
                adapter_name = lora.get("name", f"default_{random.randint(0, 1000)}") 
    
                lora_names.append(adapter_name)
                lora_weights.append(lora_weight)
                # Load LoRA weights from local path
                lora_start_time = time.time()
                pipe.load_lora_weights(
                    lora_dict, 
                    # weight_name=lora_path.split("/")[-1], 
                    adapter_name=adapter_name
                )
                pipe.fuse_lora(lora_scale = lora_weight)
                # print(time.time() - lora_start_time)
            pipe.set_adapters(lora_names, adapter_weights=lora_weights)
            # print(time.time() - start_time)
        # Prepare ControlNet configurations
    
        ip_adapters = adapters.get("ip_adapters", {})
        if ip_adapters:
            ip_scales = []
            ip_images = [] 
            ip_masks = [] 
            for ip_name, current_ip_conditions in ip_adapters.items(): 
                ip_scales.append(current_ip_conditions['ip_scale'])
    
                current_ip_images = []
                current_ip_masks = [] 
                
                for images in current_ip_conditions["images"]: 
                    ip_base64_image = images.get("image", None)
                    ip_base64_mask = images.get("mask", None)
        
                    ip_image = decode_base64_to_image(ip_base64_image)
                    ip_mask = decode_base64_to_image(ip_base64_mask)
        
                    current_ip_images.append(ip_image)
            
                ip_images.append(current_ip_images)
            
            pipe.set_ip_adapter_scale(ip_scales)
        else: 
            ip_images = None 
            ip_scales = None
            ip_masks = None
        
        controlnets = adapters.get("controlnet", {})
        controlnet_conditions = controlnets.get("conditions", [])
        condition_types = []
        if len(controlnet_conditions): 
            condition_scale = controlnets.get("conditioning_scale", 0.25)
            condition_images = {
                "openpose": 0,  
                "depth": 0, 
                "hed": 0, 
                "edges": 0, 
                "normal": 0, 
                "segment": 0, 
            }
            for controlnet_condition in controlnet_conditions:
                condition_type = controlnet_condition.get("condition_type")
                # print(f"preparing {condition_type}")
                condition_base64_image = controlnet_condition.get("condition_image")
                condition_image = decode_base64_to_image(condition_base64_image)
                condition_image = condition_image.resize((width, height))
                condition_image = np.array(condition_image)
                condition_image = processor_pipes(condition_image, condition_type)
                condition_image = condition_image.resize((width, height))
                condition_images[condition_type] = condition_image
                condition_types.append(condition_type)
        
            image_list = list(condition_images.values())
            union_control_type = torch.Tensor([bool(condition_images[ct]) for ct in condition_images])
    
            print(
                f"""
                prompt={[prompt] * num_images_per_prompt},
                image_list={image_list},
                negative_prompt={[neg_prompt] * num_images_per_prompt},
                generator={generator},
                width={width},
                height={height},
                num_inference_steps={num_steps},
                target_size={(width, height)},
                original_size={(width, height)},
                controlnet_conditioning_scale={condition_scale},
                ip_adapter_image={ip_images},
                """
            )
            
            images = pipe(
                prompt=[prompt] * num_images_per_prompt,
                image_list=image_list,
                negative_prompt=[neg_prompt] * num_images_per_prompt,
                generator=generator,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                target_size=(width, height),
                original_size=(width, height),
                controlnet_conditioning_scale=condition_scale,
                union_control=True,
                union_control_type=torch.Tensor([int(bool(image)) for image in image_list]),
                ip_adapter_image=ip_images,
            ).images
        else: 
            images = pipe(
                prompt=[prompt] * num_images_per_prompt,
                negative_prompt=[neg_prompt] * num_images_per_prompt,
                generator=generator,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                ip_adapter_image=ip_images,
            ).images
    
    # Unload LoRA weights after inference
    finally:
        if loras:
            # print("unloading lora weights...")
            pipe.unload_lora_weights()

    return images
    
# def main():
    
#     image = Image.open("test_said.jpg")
#     base64_image = encode_image_to_base64(image)
#     # image = np.array(image)

#     data = {
#         "async": True,
#         "responseFormat": "base64",
#         "sampling_parameters": {
#             "prompt": "man with big muscles, athletic body, in the city, faetastic, minecraft, pixelated, cubic",
#             "neg_prompt": "",
#             "aspect_ratio": "1:1",
#             "num_steps": 50,
#             "guidance_scale": 3.5,
#             "seed": 5,
#             "num_images_per_prompt": 1
#         },
#         "adapters": {
#             "loras": [
#                 {
#                     "name": "minecraft-1", 
#                     "url": "/home/jovyan/azizov/sdxl-api/ControlNetPlus/loras/【SDXL】MineCraft or Lego & Dataset/minecraft-mid_210380-vid_236946.safetensors",
#                     "weight": 1.0
#                 },
#                 {
#                     "name": "faetastic", 
#                     "url": "/home/jovyan/azizov/sdxl-api/ControlNetPlus/loras/faetastic.safetensors",
#                     "weight": 1.0
#                 },
#             ],
#             "controlnet": {
#                 "conditioning_scale": 0.45,
#                 "conditions": [
#                     # {
#                     #     "condition_type": "openpose",
#                     #     "condition_image": base64_image,
#                     #     "control_guidance_start": 0.0,
#                     #     "control_guidance_end": 1.0,
#                     # },
#                     # {
#                     #     "condition_type": "depth",
#                     #     "condition_image": base64_image,
#                     #     "control_guidance_start": 0.0,
#                     #     "control_guidance_end": 1.0,
#                     # },
#                     {
#                         "condition_type": "hed",
#                         "condition_image": base64_image,
#                         "control_guidance_start": 0.0,
#                         "control_guidance_end": 1.0,
#                     },
#                     {
#                         "condition_type": "edges",
#                         "condition_image": base64_image,
#                         "control_guidance_start": 0.0,
#                         "control_guidance_end": 1.0,
#                     },
#                     # {
#                     #     "condition_type": "normal",
#                     #     "condition_image": base64_image,
#                     #     "control_guidance_start": 0.0,
#                     #     "control_guidance_end": 1.0,
#                     # },
#                     # {
#                     #     "condition_type": "segment",
#                     #     "condition_image": base64_image,
#                     #     "control_guidance_start": 0.0,
#                     #     "control_guidance_end": 1.0,
#                     # },
#                 ],
#             },
#         },
#     }
    
#     image = inference(
#         controlnet_pipe if data
#         .get("adapters", {})
#         .get("controlnet", {})
#         .get("conditions", False) else pipe, 
#         data, 
#     )[0]
#     image.save("./output.png")


# if __name__ == "__main__": 
#     main()
        
    