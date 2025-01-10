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
                "prompt": ("STRING", {"multiline": True}),
                "num_images": ("INT", {"default": 2, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "subject_image": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "style_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("generated_images", "subject_prompt", "scene_prompt", "style_prompt", "prompts")
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

    def _generate_payload(self, subject_image, scene_image, style_image, prompt, session_id, num_images):
        """Generate the payload based on provided input images"""
        payload_data = {
            "json": {
                "characters": [],
                "location": None,
                "style": None,
                "pose": None,
                "additionalInput": prompt,
                "sessionId": session_id,
                "numImages": num_images
            },
            "meta": {}
        }

        if subject_image is not None and scene_image is not None and style_image is not None:
            subject_data = self._extract_image_data(subject_image, 0)
            subject_prompt = self._generate_caption(subject_data, "CHARACTER", session_id)
            subject_data["prompt"] = subject_prompt
            payload_data["json"]["characters"].append(subject_data)

            scene_data = self._extract_image_data(scene_image, 1)
            scene_prompt = self._generate_caption(scene_data, "LOCATION", session_id)
            scene_data["prompt"] = scene_prompt
            payload_data["json"]["location"] = scene_data

            style_data = self._extract_image_data(style_image, 2)
            style_prompt = self._generate_caption(style_data, "STYLE", session_id)
            style_data["prompt"] = style_prompt
            payload_data["json"]["style"] = style_data

            payload_data["meta"]["values"] = {"pose": ["undefined"]}

        elif subject_image is not None and scene_image is not None:
            subject_data = self._extract_image_data(subject_image, 0)
            subject_prompt = self._generate_caption(subject_data, "CHARACTER", session_id)
            subject_data["prompt"] = subject_prompt
            payload_data["json"]["characters"].append(subject_data)

            scene_data = self._extract_image_data(scene_image, 1)
            scene_prompt = self._generate_caption(scene_data, "LOCATION", session_id)
            scene_data["prompt"] = scene_prompt
            payload_data["json"]["location"] = scene_data

            payload_data["meta"]["values"] = {"style": ["undefined"], "pose": ["undefined"]}

        elif subject_image is not None and style_image is not None:
            subject_data = self._extract_image_data(subject_image, 0)  
            subject_prompt = self._generate_caption(subject_data, "CHARACTER", session_id)
            subject_data["prompt"] = subject_prompt
            payload_data["json"]["characters"].append(subject_data)

            style_data = self._extract_image_data(style_image, 2)
            style_prompt = self._generate_caption(style_data, "STYLE", session_id)
            style_data["prompt"] = style_prompt
            payload_data["json"]["style"] = style_data

            payload_data["meta"]["values"] = {"location": ["undefined"], "pose": ["undefined"]}

        elif scene_image is not None and style_image is not None: 
            scene_data = self._extract_image_data(scene_image, 1)
            scene_prompt = self._generate_caption(scene_data, "LOCATION", session_id)
            scene_data["prompt"] = scene_prompt
            payload_data["json"]["location"] = scene_data

            style_data = self._extract_image_data(style_image, 2)
            style_prompt = self._generate_caption(style_data, "STYLE", session_id)
            style_data["prompt"] = style_prompt  
            payload_data["json"]["style"] = style_data

            payload_data["meta"]["values"] = {"pose": ["undefined"]}

        elif subject_image is not None:
            subject_data = self._extract_image_data(subject_image, 0)
            subject_prompt = self._generate_caption(subject_data, "CHARACTER", session_id)
            subject_data["prompt"] = subject_prompt
            payload_data["json"]["characters"].append(subject_data)

            payload_data["meta"]["values"] = {"location": ["undefined"], "style": ["undefined"], "pose": ["undefined"]}

        elif scene_image is not None:
            scene_data = self._extract_image_data(scene_image, 1)
            scene_prompt = self._generate_caption(scene_data, "LOCATION", session_id)
            scene_data["prompt"] = scene_prompt
            payload_data["json"]["location"] = scene_data

            payload_data["meta"]["values"] = {"style": ["undefined"], "pose": ["undefined"]}

        elif style_image is not None:
            style_data = self._extract_image_data(style_image, 2)
            style_prompt = self._generate_caption(style_data, "STYLE", session_id)
            style_data["prompt"] = style_prompt
            payload_data["json"]["style"] = style_data

            payload_data["meta"]["values"] = {"location": ["undefined"], "pose": ["undefined"]}

        return payload_data

    def generate_image(self, prompt, subject_image=None, scene_image=None, style_image=None, num_images=2, seed=0):
        pbar = comfy.utils.ProgressBar(100)
        session_id = f";{int(time.time() * 1000)}"

        payload_data = self._generate_payload(subject_image, scene_image, style_image, prompt, session_id, num_images)

        pbar.update_absolute(30)

        try:
            storyboard_response = requests.post(
                "https://labs.google/fx/api/trpc/backbone.generateStoryBoardPrompt",
                json=payload_data,
                headers=self._get_headers(),
                cookies=self.cookies
            )
            storyboard_response.raise_for_status()
            storyboard_result = storyboard_response.json()

            if "result" in storyboard_result and "data" in storyboard_result["result"]:
                storyboard_prompt = storyboard_result["result"]["data"]["json"]
            else:
                print("Error: Unexpected response from generateStoryBoardPrompt API")
                storyboard_prompt = ""

        except requests.exceptions.RequestException as e:
            print(f"generateStoryBoardPrompt API request error: {str(e)}")
            storyboard_prompt = ""

        pbar.update_absolute(50)

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

        generated_images = torch.zeros((num_images, 512, 512, 3))
        prompts = []

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

                    prompt = image_panel.get("prompt", "")
                    prompts.append(prompt)

                if images:
                    generated_images = torch.cat(images, dim=0)
                else:
                    print("Warning: No valid images generated")
            else:
                print("Warning: No valid image panels in response")

        except requests.exceptions.RequestException as e:
            print(f"runImageFx API request error: {str(e)}")

        pbar.update_absolute(100)
        return (generated_images, 
                payload_data['json']['characters'][0]['prompt'] if payload_data['json']['characters'] else "", 
                payload_data['json']['location']['prompt'] if payload_data['json']['location'] else "", 
                payload_data['json']['style']['prompt'] if payload_data['json']['style'] else "", 
                json.dumps(prompts))


class WhiskPromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt1", "prompt2", "prompt3", "prompt4")
    FUNCTION = "process_prompts"
    CATEGORY = "comfyui-labs-google"

    def process_prompts(self, prompts):
        prompts_list = json.loads(prompts)
        prompts_list += [""] * (4 - len(prompts_list))
        return prompts_list[0], prompts_list[1], prompts_list[2], prompts_list[3]


NODE_CLASS_MAPPINGS = {
    "ComfyUI-Whisk": WhiskNode,
    "ComfyUI-Whisk-Prompts": WhiskPromptsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-Whisk": "ComfyUI-Whisk🌪️",
    "ComfyUI-Whisk-Prompts": "ComfyUI-Whisk-Prompts🌪️"
}
