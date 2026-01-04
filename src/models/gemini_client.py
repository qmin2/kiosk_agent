from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
import sys
from PIL import Image

# Ensure the repository root is on sys.path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3] # 02_Pseudo_Lab 폴더
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from kiosk_agent.src.config import ModelConfig
from kiosk_agent.src.models.base import BaseModelClient
from kiosk_agent.src.utils.schema import GUI_OUTPUT

class GeminiClient(BaseModelClient):
    """Calls the Gemini SDK vision models."""

    def __init__(self, config: ModelConfig):
        if not config.gemini_api_key:
            raise ValueError("Set ModelConfig.gemini_api_key when provider='gemini'.")
        super().__init__(config)
        self._client = genai.Client(api_key=config.gemini_api_key)

    def generate(self, instruction: str, image: Image.Image) -> str:
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            system_instruction = self.config.system_prompt,
            response_mime_type="application/json",
            response_json_schema= GUI_OUTPUT.model_json_schema(),
        )
        contents = self._build_parts(instruction, image)
        response = self._client.models.generate_content(
            model=self.config.gemini_model,
            contents=contents,
            config=config,
        )
        # breakpoint()
        output_text = (response.text or "").strip()
        if not output_text:
            raise RuntimeError("Gemini returned an empty response.")
        
        width, height = image.size
        print(response.text)
        dict_response = json.loads(response.text)

        converted_bounding_boxes = []
        abs_y1 = int(dict_response["box_2d"][0]/1000 * height)
        abs_x1 = int(dict_response["box_2d"][1]/1000 * width)
        abs_y2 = int(dict_response["box_2d"][2]/1000 * height)
        abs_x2 = int(dict_response["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_y1, abs_x1, abs_y2, abs_x2])

        print(f"Image size:  y: {height}, x: {width}")
        print("Converted Bounding boxes:", converted_bounding_boxes)

        mid_y = (abs_y1 + abs_y2)/2
        mid_x = (abs_x1 + abs_x2)/2
        print("Converted Mid point:", [mid_x, mid_y])
        print("thought:", dict_response["thought"])
        # return response
        return response.text, width, height

    # Function to encode the image
    def encode_image(self, image_path): # from https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded
        with open(image_path, "rb") as image_file:
            return image_file.read() # image bytes
    
    # def _build_system_instruction(self) -> str:
    #     schema_hint = json.dumps(self.action_schema, ensure_ascii=False)
    #     gui_schema_hint = json.dumps(GUI_OUTPUT.model_json_schema(), ensure_ascii=False)
    #     return (
    #         f"{self.config.system_prompt.strip()}\n\n"
    #         f"Respond with JSON that strictly follows this action schema:\n{schema_hint}\n\n"
    #         f"For reference, here is the simplified GUI schema:\n{gui_schema_hint}"
    #     )

    def _build_parts(self, instruction: str, image: Optional[Image.Image]) -> List[types.Part]:
        # parts: List[types.Part] = [types.Part.from_text(instruction)]
        parts = []
        image_part = image
        # image_part = self._build_image_part(image)
        if image_part is not None:
            parts.append(image_part)
        
        parts.append(instruction)
        return parts

    @staticmethod
    def _build_image_part(image: Optional[Image.Image]) -> Optional[types.Part]:
        if image is None:
            return None
        return types.Part.from_bytes(
            data=image,
            mime_type="image/png",
        )


if __name__ == "__main__":
    from dataclasses import dataclass
    import os
    from pathlib import Path
    from kiosk_agent.src.prompts.prompts import GEMINI_SYSTEM_PROMPT

    import time

    @dataclass
    class ModelConfig:
        """Configuration shared by all model providers."""

        provider: Literal["chatgpt", "gemini", "local_vllm"] = "gemini"
        system_prompt: str = GEMINI_SYSTEM_PROMPT
        
        # ChatGPT / OpenAI
        openai_model: str = "gpt-4o-mini"
        openai_api_key: str = os.getenv("OPENAI_API_KEY")
        openai_api_base: Optional[str] = None
        # Gemini
        gemini_model: str = "gemini-2.5-flash"
        gemini_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        # Local vLLM (OpenAI-compatible HTTP server)
        vllm_base_url: str = "http://localhost:8000"
        vllm_model_name: str = "AgentCPM-GUI"
        vllm_api_key: Optional[str] = None



    client= GeminiClient(config=ModelConfig())
    image_path = "/Users/qmin2/02_Pseudo_Lab/kiosk_agent/screenshots/test1.jpeg" # sample absolute path
    image = Image.open(image_path)
    # adb_config = ADBConfig()
    # ADBController = ADBController(adb_config)

    for i in range(1):
        responses_dict = client.generate("빅맥 세트를 주문하고 싶고, 콜라는 레귤러, 감자튀김도 미디엄 사이즈로 골라줘", image)

        converted_bounding_boxes = []
        response, width, height = responses_dict

        # print(response)
        # print(type(response))
        response = json.loads(response)
        abs_y1 = int(response["box_2d"][0]/1000 * height)
        abs_x1 = int(response["box_2d"][1]/1000 * width)
        abs_y2 = int(response["box_2d"][2]/1000 * height)
        abs_x2 = int(response["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_y1, abs_x1, abs_y2, abs_x2])

        print(f"Image size:  y: {height}, x: {width}")
        print("Converted Bounding boxes:", converted_bounding_boxes)

        mid_y = (abs_y1 + abs_y2)/2
        mid_x = (abs_x1 + abs_x2)/2
        print("Converted Mid point:", [mid_x, mid_y])
        print("thought:", response["thought"])

        # adb_result = ADBController.screenshot()

        # image_path = adb_result[-1]

        # ADBController.tap(mid_x, mid_y)
        # time.sleep(1.5)
