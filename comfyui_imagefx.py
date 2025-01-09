import os
import json
import base64
import requests
from io import BytesIO
import numpy as np
import torch
import time
from PIL import Image, UnidentifiedImageError
from typing import List, Union, Tuple
import comfy.utils
from .utils import pil2tensor, tensor2pil
import chardet

class ComfyUIImageFxNode:
    def __init__(self):
        self._initialize_auth()
        self.aspect_ratio_display = {
            "1:1 (Square)": "IMAGE_ASPECT_RATIO_SQUARE",
            "9:16 (Portrait)": "IMAGE_ASPECT_RATIO_PORTRAIT",
            "16:9 (Landscape)": "IMAGE_ASPECT_RATIO_LANDSCAPE",
            "3:4 (Portrait)": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
            "4:3 (Landscape)": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE"
        }

    def _initialize_auth(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(current_dir, 'googel.json')

            with open(config_file_path, 'rb') as file:
                content = file.read()
                encoding = chardet.detect(content)['encoding']

            with open(config_file_path, 'r', encoding=encoding) as f:
                self.auth_config = json.load(f)
                
            self.access_token = self.auth_config.get('access_token')
            self.user = self.auth_config.get('user', {})
            self.expires = self.auth_config.get('expires')
            self.cookies = {cookie['name']: cookie['value'] for cookie in self.auth_config.get('cookies', [])}

            if not self.access_token:
                raise ValueError("Access token not found in googel.json")

        except Exception as e:
            print(f"Authentication initialization error: {str(e)}")
            raise

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = [
            "1:1 (Square)", 
            "9:16 (Portrait)",
            "16:9 (Landscape)", 
            "3:4 (Portrait)",
            "4:3 (Landscape)"
        ]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "aspect_ratio": (aspect_ratios, {"default": "16:9 (Landscape)"}),
                "num_images": ("INT", {"default": 4, "min": 1, "max": 4})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "seed")
    FUNCTION = "generate_image"
    CATEGORY = "comfyui-labs-google"

    def _get_api_aspect_ratio(self, display_ratio: str) -> str:
        """Convert display aspect ratio to API format"""
        return self.aspect_ratio_display.get(display_ratio, "IMAGE_ASPECT_RATIO_LANDSCAPE")

    def generate_image(self, prompt: str, seed: int, aspect_ratio: str, num_images: int = 4) -> Tuple[torch.Tensor, str]:
        pbar = comfy.utils.ProgressBar(100)
        session_id = f";{int(time.time() * 1000)}"
        api_aspect_ratio = self._get_api_aspect_ratio(aspect_ratio)

        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "authorization": f"Bearer {self.access_token}",
            "content-type": "application/json",
            "origin": "https://labs.google",
            "referer": "https://labs.google/",
            "sec-ch-ua": '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

        json_data = {
            "userInput": {
                "candidatesCount": num_images,
                "prompts": [prompt],
                "isExpandedPrompt": False,
                "seed": seed % 1000000
            },
            "clientContext": {
                "sessionId": session_id,
                "tool": "IMAGE_FX"
            },
            "aspectRatio": api_aspect_ratio,
            "modelInput": {
                "modelNameType": "IMAGEN_3_1"
            }
        }

        pbar.update_absolute(20)

        try:
            cleaned_headers = {k: v for k, v in headers.items() if not k.startswith(':')}
            
            response = requests.post(
                "https://aisandbox-pa.googleapis.com/v1:runImageFx",
                json=json_data,
                headers=cleaned_headers,
                cookies=self.cookies
            )
            
            response.raise_for_status()
            result = response.json()

            pbar.update_absolute(50)

            if "imagePanels" in result:
                image_panel = result["imagePanels"][0]
                images = []
                response_seed = None
                
                for img_data in image_panel["generatedImages"]:
                    try:
                        encoded_image = img_data["encodedImage"]
                        if response_seed is None:
                            response_seed = img_data.get("seed", seed)
                        
                        # Decode base64 to bytes
                        if "," in encoded_image:
                            encoded_image = encoded_image.split(",", 1)[1]
                        image_bytes = base64.b64decode(encoded_image)
                        
                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(image_bytes))
                        img_tensor = pil2tensor(pil_image)
                        images.append(img_tensor)
                        
                        pbar.update_absolute(50 + (40 * len(images) // num_images))
                        
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                
                if images:
                    # Combine all images into a single tensor with shape [N, H, W, 3]
                    combined_tensor = torch.cat(images, dim=0)
                    pbar.update_absolute(100)
                    return (combined_tensor, str(response_seed))
                
            # Return empty tensor with correct shape if no valid images
            pbar.update_absolute(100)
            return (torch.zeros((num_images, 512, 512, 3)), str(seed))

        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            return (torch.zeros((num_images, 512, 512, 3)), str(seed))
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return (torch.zeros((num_images, 512, 512, 3)), str(seed))


NODE_CLASS_MAPPINGS = {
    "ComfyUI-ImageFx": ComfyUIImageFxNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-ImageFx": "ComfyUI-ImageFx🖼️"
}
