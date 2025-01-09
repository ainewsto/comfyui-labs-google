import os
import json
import base64
import requests
from io import BytesIO
import numpy as np
import torch
import uuid
import time
from PIL import Image, UnidentifiedImageError
from typing import List, Union
import comfy.utils
from .utils import pil2tensor, tensor2pil
import chardet

class WhiskNode:
    def __init__(self):
        self._initialize_auth()

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
        return {
            "required": {
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),  
                "style_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "num_images": ("INT", {"default": 2, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("generated_images", "subject_prompt", "scene_prompt", "style_prompt", "seed") 
    FUNCTION = "generate_image"
    CATEGORY = "comfyui-labs-google"

    def _get_headers(self):
        return {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "authorization": f"Bearer {self.access_token}",
            "content-type": "application/json",
            "origin": "https://labs.google",
            "referer": "https://labs.google/fx/zh/tools/whisk",
            "sec-ch-ua": '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

    def _generate_caption(self, image_data, category, session_id):
        """Generate caption for a single image"""
        headers = self._get_headers()
        
        caption_json_data = {
            "json": {
                "category": category,
                "imageBase64": image_data["base64Image"],
                "sessionId": session_id
            }
        }
        
        try:
            caption_response = requests.post(
                "https://labs.google/fx/api/trpc/backbone.generateCaption",
                json=caption_json_data,
                headers=headers,
                cookies=self.cookies
            )
            caption_response.raise_for_status()
            
            result = caption_response.json()
            if "result" in result and "data" in result["result"] and "json" in result["result"]["data"]:
                return result["result"]["data"]["json"]
            else:
                print(f"Error: Unexpected response format from generateCaption API for {category}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"generateCaption API request error for {category}: {str(e)}")
            return ""

    def _extract_image_data(self, image_tensor, index):
        """Extract image data and convert to base64"""
        pil_image = tensor2pil(image_tensor)[0]  # Convert tensor to PIL image
        image_bytes = self._pil_to_bytes(pil_image)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "imageId": f"image-{uuid.uuid4()}",
            "category": ["CHARACTER", "LOCATION", "STYLE"][index],
            "isPlaceholder": False,
            "base64Image": f"data:image/jpeg;base64,{base64_image}",
            "index": index,
            "isUploading": False,
            "isLoading": False,
            "isSelected": True,
            "prompt": ""
        }

    def _pil_to_bytes(self, image):
        """Convert PIL image to bytes"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    def generate_image(self, subject_image, scene_image, style_image, prompt, num_images=2, seed=0):
        pbar = comfy.utils.ProgressBar(100)
        session_id = f";{int(time.time() * 1000)}"
        
        # Extract image data for each input
        image_data = [
            self._extract_image_data(subject_image, 0),
            self._extract_image_data(scene_image, 1),
            self._extract_image_data(style_image, 2)
        ]
        pbar.update_absolute(10)
        
        # Generate captions for each image
        subject_prompt = self._generate_caption(image_data[0], "CHARACTER", session_id)
        scene_prompt = self._generate_caption(image_data[1], "LOCATION", session_id)
        style_prompt = self._generate_caption(image_data[2], "STYLE", session_id)
        pbar.update_absolute(30)
        
        # Update prompts in image data
        image_data[0]["prompt"] = subject_prompt
        image_data[1]["prompt"] = scene_prompt
        image_data[2]["prompt"] = style_prompt
        
        # Prepare storyboard request
        storyboard_json_data = {
            "json": {
                "characters": [img_data for img_data in image_data if img_data["category"] == "CHARACTER"],
                "location": next((img_data for img_data in image_data if img_data["category"] == "LOCATION"), None),
                "style": next((img_data for img_data in image_data if img_data["category"] == "STYLE"), None),
                "pose": None,
                "additionalInput": prompt,
                "sessionId": session_id,
                "numImages": num_images
            },
            "meta": {
                "values": {"pose": ["undefined"]}
            }
        }
        pbar.update_absolute(50)
        
        try:
            storyboard_response = requests.post(
                "https://labs.google/fx/api/trpc/backbone.generateStoryBoardPrompt",
                json=storyboard_json_data,
                headers=self._get_headers(),
                cookies=self.cookies
            )
            storyboard_response.raise_for_status()
            storyboard_result = storyboard_response.json()
            
            if "result" in storyboard_result and "data" in storyboard_result["result"]:
                storyboard_prompt = storyboard_result["result"]["data"]["json"]
            else:
                print("Error: Unexpected response from generateStoryBoardPrompt API")
                return torch.zeros((num_images, 512, 512, 3)), subject_prompt, scene_prompt, style_prompt
        
        except requests.exceptions.RequestException as e:
            print(f"generateStoryBoardPrompt API request error: {str(e)}")
            return torch.zeros((num_images, 512, 512, 3)), subject_prompt, scene_prompt, style_prompt
        
        pbar.update_absolute(70)
            
        # Call runImageFx API with the generated storyboard prompt
        imagefx_json_data = {
            "userInput": {
                "candidatesCount": num_images,
                "prompts": [storyboard_prompt],
                "isExpandedPrompt": False,
                "seed": seed % 2147483647
            },
            "clientContext": {
                "sessionId": session_id,
                "tool": "BACKBONE"
            },
            "aspectRatio": "IMAGE_ASPECT_RATIO_LANDSCAPE",
            "modelInput": {
                "modelNameType": "IMAGEN_3_1"
            }
        }
        
        try:
            imagefx_response = requests.post(
                "https://aisandbox-pa.googleapis.com/v1:runImageFx",
                json=imagefx_json_data,
                headers=self._get_headers(),
                cookies=self.cookies
            )
            imagefx_response.raise_for_status()
            imagefx_result = imagefx_response.json()
            
            if "imagePanels" in imagefx_result:
                image_panel = imagefx_result["imagePanels"][0]
                images = []
                
                for img_data in image_panel["generatedImages"]:
                    encoded_image = img_data["encodedImage"]
                    if "," in encoded_image:
                        encoded_image = encoded_image.split(",", 1)[1]
                    
                    image_bytes = base64.b64decode(encoded_image)
                    pil_image = Image.open(BytesIO(image_bytes))
                    img_tensor = pil2tensor(pil_image)
                    images.append(img_tensor)
                
                if images:
                    generated_images = torch.cat(images, dim=0)
                    pbar.update_absolute(100)
                    return generated_images, subject_prompt, scene_prompt, style_prompt, seed
            
            print("Error: No valid images generated")
            return torch.zeros((num_images, 512, 512, 3)), subject_prompt, scene_prompt, style_prompt, seed
                
        except requests.exceptions.RequestException as e:
            print(f"runImageFx API request error: {str(e)}")
            return torch.zeros((num_images, 512, 512, 3)), subject_prompt, scene_prompt, style_prompt, seed


NODE_CLASS_MAPPINGS = {
    "ComfyUI-Whisk": WhiskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-Whisk": "ComfyUI-Whisk🌪️"
}
