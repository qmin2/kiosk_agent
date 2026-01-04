from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kiosk_agent.src.config import ADBConfig, AgentConfig, ModelConfig, ScreenshotConfig, STTConfig
from kiosk_agent.src.langgraph_kiosk_agent import KioskAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the kiosk GUI agent for a single step.")
    parser.add_argument("instruction", nargs="?", default=None, help="User request the agent should fulfil.")
    parser.add_argument("--use-stt", action="store_true", help="Use Google STT to get instruction from microphone.")
    parser.add_argument("--stt-file", type=str, default=None, help="Path to WAV file for STT (instead of microphone).")
    parser.add_argument("--stt-timeout", type=float, default=10.0, help="STT recording timeout in seconds.")
    parser.add_argument("--stt-streaming", action="store_true", help="Use streaming STT (stops on final result).")
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

    # Determine STT mode
    # --use-stt: STT 활성화 (필수)
    # --stt-file: 있으면 파일 모드, 없으면 마이크 모드
    # --stt-streaming: 마이크 모드일 때 스트리밍 사용
    stt_enabled = args.use_stt

    if args.stt_file:
        stt_mode = "file"
    elif args.stt_streaming:
        stt_mode = "streaming"
    else:
        stt_mode = "microphone"

    # Build STT config
    stt_config = STTConfig(
        enabled=stt_enabled,
        mode=stt_mode,
        file_path=args.stt_file,
        timeout_seconds=args.stt_timeout,
    )

    # Get instruction from command line (STT will be handled in the graph if enabled)
    instruction = args.instruction

    # If STT is not enabled, instruction is required
    if not stt_enabled and not instruction:
        print("[ERROR] instruction이 필요합니다. 텍스트로 입력하거나 --use-stt 옵션을 사용하세요.")
        return

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
        stt=stt_config,
        schema_path=Path(args.schema_path),
    )

    agent = KioskAgent(config)
    result = agent.forward(instruction)
    # print("Status:", result.status)
    # print("Thought:", result.thought)
    # # print("ADB commands:")
    # for command in result.adb_commands:
    #     print(" ", " ".join(command))
    # if result.screenshot_path:
    #     print("Screenshot saved to:", result.screenshot_path)


if __name__ == "__main__":
    main()
