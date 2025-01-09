from .comfyui_whisk import NODE_CLASS_MAPPINGS as WHISK_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as WHISK_NODE_DISPLAY_NAME_MAPPINGS
from .comfyui_imagefx import NODE_CLASS_MAPPINGS as COMFYUI_IMAGEFX_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as COMFYUI_IMAGEFX_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**WHISK_NODE_CLASS_MAPPINGS, **COMFYUI_IMAGEFX_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**WHISK_NODE_DISPLAY_NAME_MAPPINGS, **COMFYUI_IMAGEFX_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
