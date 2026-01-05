from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from kiosk_agent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
# Parent of project root for broader context if needed
PARENT_ROOT = Path(__file__).resolve().parents[2]
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from copilotkit.sdk import CopilotKitSDK
# from copilotkit.integrations.langgraph import LangGraphAgent
from copilotkit import LangGraphAGUIAgent

from kiosk_agent.src.config import AgentConfig, ADBConfig, ModelConfig, ScreenshotConfig
from kiosk_agent.src.langgraph_kiosk_agent import KioskAgent

# 1. Initialize the Kiosk Agent and its LangGraph
# We'll use a local 'screenshots' directory in the project root
screenshots_dir = REPO_ROOT / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)

screenshot_config = ScreenshotConfig(output_dir=screenshots_dir)
model_config = ModelConfig(provider="gemini")
adb_config = ADBConfig()
config = AgentConfig(
    screenshot=screenshot_config,
    model=model_config,
    adb=adb_config,
)

# KioskAgent internally builds the graph
agent = KioskAgent(config)
graph = agent._build_graph()


# 2. Setup FastAPI
app = FastAPI()

# Enable CORS so the React frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount screenshots directory to serve images
# Note: Ensure this matches where the agent saves screenshots
screenshots_dir = REPO_ROOT / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=str(screenshots_dir)), name="screenshots")



# 3. Setup CopilotKit SDK with the LangGraph agent
sdk = CopilotKitSDK(
    agents=[
        LangGraphAGUIAgent(
            name="kiosk_agent",
            description="An agent that controls an Android kiosk via ADB.",
            graph=graph,
        )
    ]
)

@app.post("/copilotkit")
async def copilotkit_endpoint(request: Request):
    """The main endpoint for CopilotKit communication."""
    return await sdk.handle_request(request)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
