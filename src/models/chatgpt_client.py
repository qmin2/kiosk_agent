from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple

# Ensure the repository root is on sys.path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from PIL import Image

from kiosk_agent.src.config import ModelConfig
from kiosk_agent.src.models.base import BaseModelClient


from pydantic import BaseModel, Field

class GUI_OUTPUT(BaseModel):
    thought: str = Field(description="Agent reasoning (not executed)")
    action: Literal['CLICK', 'LONG_CLICK', 'SWIPE', 'INPUT', 'BACK', 'HOME'] = Field(description="Type of GUI action to perform.")
    position: List[float] = Field(
        description="Normalized (0~1) screen coordinate [x,y] for action.",
        min_items=2,
        max_items=2,
    )                                                                  

class ChatGPTClient(BaseModelClient):
    """Calls OpenAI's ChatGPT vision models."""

    def __init__(self, config: ModelConfig):
        if not config.openai_api_key:
            raise ValueError("Set ModelConfig.openai_api_key when provider='chatgpt'.")
        super().__init__(config)
        self._client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_api_base)

    def generate(self, instruction: str, image: Image.Image) -> str:
        encoded_image = self._encode_image(image)

        # Use standard chat.completions API for OpenRouter compatibility
        response = self._client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": self.config.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}",
                            }
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content
        print(result)
        return result

    # Function to encode the image
    def encode_image(self, image_path): # from https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _encode_image(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_completion(self, content: Any) -> str: # 
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content)


if __name__ == "__main__":
    from dataclasses import dataclass
    import os
    from kiosk_agent.utils.action_schema import load_action_schema
    from pathlib import Path
    action_schema = load_action_schema(Path("/Users/qmin2/Desktop/kiosk_agent/utils/schema/schema_showui.json"))
    
    @dataclass
    class ModelConfig:
        """Configuration shared by all model providers."""

        provider: Literal["chatgpt", "gemini", "local_vllm"] = "local_vllm"
        system_prompt: str = (
            """Based on the screenshot of the page, I give a text description and you give its corresponding location. 
            The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."""

            # "You are an Android kiosk assistant. Analyze the screenshot and respond "
            # "with a JSON action that follows the shared action schema."
        )
        # temperature: float = 0.1
        # top_p: float = 0.3
        # ChatGPT / OpenAI
        openai_model: str = "gpt-4o-mini"
        openai_api_key: str = os.getenv("OPENAI_API_KEY")
        openai_api_base: Optional[str] = None
        # Gemini
        gemini_model: str = "gemini-2.5-flash"
        gemini_api_key: Optional[str] = None
        # Local vLLM (OpenAI-compatible HTTP server)
        vllm_base_url: str = "http://localhost:8000"
        vllm_model_name: str = "AgentCPM-GUI"
        vllm_api_key: Optional[str] = None

    client= ChatGPTClient(config=ModelConfig(), action_schema=action_schema)
    client.generate("테이크 아웃 버튼 눌러줘", None)
