from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import os
from kiosk_agent.src.prompts.vlm_system_prompt import VLM_GEMINI_SYSTEM_PROMPT

@dataclass
class ScreenshotConfig: ## 이게 굳이 있어야 하나?
    """Configuration for how screenshots are captured from the kiosk device.""" 

    adb_path: str = "adb"
    device_id: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("./screenshots"))
    keep_last_n: int = 10


@dataclass
class ModelConfig:
    """Configuration shared by all model providers."""

    provider: Literal["chatgpt", "gemini", "local_vllm"] = "local_vllm"
    system_prompt: str = VLM_GEMINI_SYSTEM_PROMPT
    # temperature: float = 0.1
    # top_p: float = 0.3
    # ChatGPT / OpenAI
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_api_base: Optional[str] = None
    # Gemini
    gemini_model: str = "gemini-3-flash-preview" # "gemini-3-pro-preview" # "gemini-3-flash-preview"
    gemini_api_key: str =  os.getenv("GOOGLE_API_KEY")
    # Local vLLM (OpenAI-compatible HTTP server)
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "AgentCPM-GUI"
    vllm_api_key: Optional[str] = None


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text input."""
    enabled: bool = False
    mode: Literal["file", "microphone", "streaming"] = "microphone"
    file_path: Optional[str] = None
    language_code: str = "ko-KR"
    sample_rate_hz: int = 16000
    timeout_seconds: float = 10.0


@dataclass
class ADBConfig:
    """Configuration for translating structured actions into concrete adb commands."""
    adb_path: str = "adb"
    device_id: Optional[str] = None
    default_swipe_duration_ms: int = 300
    steps: int = 1
    screenshot_abs_path: str = "./screenshots"


@dataclass
class AgentConfig:
    """Top level configuration bundle."""
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adb: ADBConfig = field(default_factory=ADBConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    schema_path: Path = Path("./utils/deprecated_schema/schema_showui.json")
