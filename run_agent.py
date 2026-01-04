from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
load_dotenv()

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
    parser.add_argument("--screenshot-abs-path", default="artifacts/screens", help="Absolute path for screenshots on device.")
    parser.add_argument("--schema-path", default="src/utils/schema/schema_showui.json", help="Path to schema JSON file.")
    parser.add_argument("--keep-last-n", type=int, default=10, help="Number of screenshots to keep.")
    parser.add_argument("--swipe-duration-ms", type=int, default=300, help="Default swipe duration in ms.")
    # ChatGPT / OpenRouter
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-base", default=None, help="Custom API base URL (e.g., https://openrouter.ai/api/v1)")
    # Gemini
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview")
    # Local vLLM
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--vllm-model-name", default="AgentCPM-GUI")
    parser.add_argument("--vllm-api-key", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    screenshot_config = ScreenshotConfig(
        adb_path=args.adb_path,
        device_id=args.device_id,
        output_dir=Path(args.screenshot_dir),
        keep_last_n=args.keep_last_n,
    )
    model_config = ModelConfig(
        provider=args.provider,
        openai_model=args.openai_model,
        openai_api_base=args.openai_api_base,
        gemini_model=args.gemini_model,
        vllm_base_url=args.vllm_base_url,
        vllm_model_name=args.vllm_model_name,
        vllm_api_key=args.vllm_api_key,
    )
    adb_config = ADBConfig(
        adb_path=args.adb_path,
        device_id=args.device_id,
        default_swipe_duration_ms=args.swipe_duration_ms,
        screenshot_abs_path=args.screenshot_abs_path,
    )
    config = AgentConfig(
        screenshot=screenshot_config,
        model=model_config,
        adb=adb_config,
        schema_path=Path(args.schema_path),
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
