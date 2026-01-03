from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiosk_agent.src.config import ADBConfig, AgentConfig, ModelConfig, ScreenshotConfig
from kiosk_agent.src.langgraph_kiosk_agent import KioskAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kiosk GUI agent for a single step.")
    parser.add_argument("instruction", help="User request the agent should fulfil.")
    parser.add_argument("--provider", choices=["chatgpt", "gemini", "local_vllm"], default="gemini")
    parser.add_argument("--device-id", default=None, help="adb device serial if multiple devices are connected.")
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--screenshot-dir", default="artifacts/screens")
    # ChatGPT
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    # Gemini
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview")
    # Local vLLM
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--vllm-model-name", default="GUI_agent")
    # parser.add_argument("--vllm-api-key", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    screenshot_config = ScreenshotConfig(
        adb_path=args.adb_path,
        device_id=args.device_id,
        # output_dir=Path(args.screenshot_dir),
    )
    model_config = ModelConfig(
        provider=args.provider,
        # openai_model=args.openai_model,
        # gemini_model=args.gemini_model,
        # vllm_base_url=args.vllm_base_url,
        # vllm_model_name=args.vllm_model_name,
    )
    adb_config = ADBConfig(
        adb_path=args.adb_path,
        device_id=args.device_id,
    )
    config = AgentConfig(
        screenshot=screenshot_config,
        model=model_config,
        adb=adb_config,
    )
    
    agent = KioskAgent(config)
    result = agent.forward(args.instruction)
    # print("Status:", result.status)
    # print("Thought:", result.thought)
    # # print("ADB commands:")
    # for command in result.adb_commands:
    #     print(" ", " ".join(command))
    # if result.screenshot_path:
    #     print("Screenshot saved to:", result.screenshot_path)


if __name__ == "__main__":
    main()
