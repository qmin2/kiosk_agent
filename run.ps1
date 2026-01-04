[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================
# Input Mode Selection:
# ============================================================
# --use-stt: STT 활성화 (필수)
# --stt-file: 있으면 WAV 파일 모드, 없으면 마이크 모드
# --stt-streaming: 마이크 모드일 때 스트리밍 사용
#
# 예시:
# 1. 텍스트 입력: python run_agent.py "주문해줘" --provider gemini ...
# 2. WAV 파일:    python run_agent.py --use-stt --stt-file "./audio.wav" ...
# 3. 마이크:      python run_agent.py --use-stt --stt-timeout 10.0 ...
# 4. 스트리밍:    python run_agent.py --use-stt --stt-streaming ...
# ============================================================

# STT + WAV 파일 모드
python run_agent.py `
    --use-stt `
    --stt-file "./src/prompts/output_hamburger.wav" `
    --provider gemini `
    --adb-path adb `
    --screenshot-dir ./screenshot `
    --screenshot-abs-path ./screenshot `
    --schema-path src/utils/schema/schema_showui.json `
    --keep-last-n 10 `
    --swipe-duration-ms 300 `
    --stt-timeout 10.0 `
    --openai-model anthropic/claude-3.5-sonnet `
    --openai-api-base https://openrouter.ai/api/v1 `
    --gemini-model gemini-3-flash-preview `
    --vllm-base-url http://localhost:8000 `
    --vllm-model-name AgentCPM-GUI
