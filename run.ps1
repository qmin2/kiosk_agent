[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$instruction = "콰트로치즈와퍼 세트를 가장 가까운 매장에서 먹고가기로 주문하기 위해 장바구니에 담아줘"

# STT options (uncomment one to use):
# --use-stt                      # Use microphone (records for stt-timeout seconds)
# --stt-file "./audio/order.wav" # Use WAV file
# --stt-streaming                # Use streaming mode (stops when speech ends)
# --stt-timeout 10.0             # Recording timeout in seconds (default: 10)

python run_agent.py $instruction `
    --provider gemini `
    --adb-path adb `
    --screenshot-dir ./screenshot `
    --screenshot-abs-path ./screenshot `
    --schema-path src/utils/schema/schema_showui.json `
    --keep-last-n 10 `
    --swipe-duration-ms 300 `
    --stt-file "./src/prompts/output_hamburger.wav" `
    --stt-timeout 10.0 `
    --openai-model anthropic/claude-3.5-sonnet `
    --openai-api-base https://openrouter.ai/api/v1 `
    --gemini-model gemini-3-flash-preview `
    --vllm-base-url http://localhost:8000 `
    --vllm-model-name AgentCPM-GUI
