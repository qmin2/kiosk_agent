from __future__ import annotations

import json
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, TypedDict, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
# from langgraph.checkpoint.memory import MemorySaver

from kiosk_agent.src.config import AgentConfig
from kiosk_agent.src.control.adb import ADBController
from kiosk_agent.src.control.translator import ActionTranslator
from kiosk_agent.src.models.base import BaseModelClient, ModelAction
from kiosk_agent.src.models.chatgpt_client import ChatGPTClient
from kiosk_agent.src.models.gemini_client import GeminiClient
from kiosk_agent.src.models.local_vllm_client import LocalVLLMClient
from kiosk_agent.src.perception.screenshot import AndroidScreenshotter, ScreenshotResult
from kiosk_agent.src.prompts.vlm_system_prompt import VLM_GEMINI_USER_PROMPT
        
from kiosk_agent.src.utils.utils import (
    parse_analysis_response, 
    compute_difference_from_paths,
    build_analysis_prompt,
    build_result,
    compute_screen_id,
)
from kiosk_agent.src.utils.types import (
    HistoryEntry, 
    AgentState, 
    AgentStepResult, 
    AgentStreamEvent
)

class LangGraphKioskAgent:
    """High level facade that glues together perception, reasoning and control."""

    def __init__(self, config: AgentConfig, dry_run: bool = False):
        self.config = config
        self.current_screen_path = None
        self._current_screen_image: Optional[Image.Image] = None
        self.max_iterations = 20
        self.progress_threshold = 0.02
        self.human_review_trigger = 2  # number of consecutive non-progress steps before escalations
        self.graph = None
        # self.checkpointer = MemorySaver()
        
        provider = config.model.provider

        if provider == "chatgpt":
            model_client = ChatGPTClient(config.model)
        elif provider == "gemini":
            model_client = GeminiClient(config.model)
        elif provider == "local_vllm":
            model_client = LocalVLLMClient(config.model)
        else:
            raise ValueError(f"Unsupported model provider '{provider}'.")
        adb_controller = ADBController(config.adb)
        self.translator = ActionTranslator(adb_controller, config.adb)
        self.model_client = model_client
        self.analysis_client = model_client
        self.screenshotter = AndroidScreenshotter(config.screenshot)
        self.graph = None

    def _capture_screen(self) -> ScreenshotResult:
        """Capture the current kiosk screen and cache it for future use."""
        screenshot = self.screenshotter.capture(save=True)
        self.current_screen_path = screenshot.path
        self._current_screen_image = screenshot.image
        return screenshot

    def _get_screen(self, *, refresh: bool = False) -> ScreenshotResult:
        """Return the cached screenshot, refreshing it from the device if needed."""
        if refresh or self._current_screen_image is None or self.current_screen_path is None:
            return self._capture_screen()
        return ScreenshotResult(image=self._current_screen_image, path=self.current_screen_path)

    def forward(
        self,
        instruction: str,
        previous_state: Optional[AgentState] = None,
        thread_id: Optional[str] = None,
    ) -> AgentStepResult:
        """Run the langgraph-based perception → reasoning → control workflow."""
        
        # Prepare graph and initial state
        graph, initial_state = self.prepare_workflow(instruction, previous_state)
        
        # Set up configuration
        config = {"recursion_limit": 100}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        final_state = graph.invoke(initial_state, config)
        return build_result(final_state)

    def workflow_tracer(self, ):
        # TODO: 오류 발생시 이전 step으로 되돌아가기
        pass

    def init_state(self, instruction: str) -> AgentState:
        self._capture_screen()
        initial_state: AgentState = {
            "instruction": instruction,
            "iteration": 0,
            "status": "init",
            "route": "vlm",
            "history": [],
            "human_decision": None,
            "difference": None,
            "progress": None,
            "last_adb_commands": [],
            "last_iteration_id": -1,
            "current_screen_id": None,
        }
        screen_id = compute_screen_id(self.current_screen_path)
        initial_state["current_screen_id"] = screen_id
        return initial_state

    def prepare_workflow(
        self, instruction: str, previous_state: Optional[AgentState] = None
    ) -> tuple[CompiledStateGraph, AgentState]:
        """Return the compiled workflow and its initialized state."""

        if previous_state:
            initial_state = previous_state
            # If we are resuming, ensure capturing the current screen if needed,
            # but usually the previous state might have the context.
            # To be safe, we might want to refresh the screen if it's a new session.
            # self._capture_screen() # Optional: decide if resume implies new screenshot
        else:
            initial_state = self.init_state(instruction)

        self.graph = self._build_graph()
        return self.graph, initial_state

    # def stream_workflow(
    #     self,
    #     instruction: str,
    #     *,
    #     stream_mode: str = "values",
    #     config: Optional[Dict[str, Any]] = None,
    # ) -> Iterator[AgentStreamEvent]:
    #     """Yield structured stream events for each LangGraph node transition."""
    #     print("INSIDE")

    #     breakpoint()
    #     graph, initial_state = self.prepare_workflow(instruction)
    #     latest_state: Dict[str, Any] = copy.deepcopy(initial_state)
    #     breakpoint()
    #     for chunk in graph.stream(initial_state, config=config, stream_mode=stream_mode):
    #         breakpoint()
    #         stage, update = self._extract_stage_update(chunk)
    #         if stage is None:
    #             continue
    #         latest_state.update(update)
    #         # requires_human = bool(latest_state.get("requires_human_input")) # TODO: 복잡한 query일때 hitl 추가
    #         requires_human = False
    #         message = self._format_stage_message(stage, latest_state, requires_human)
    #         yield AgentStreamEvent(
    #             stage=stage,
    #             state=cast(AgentState, copy.deepcopy(latest_state)),
    #             message=message,
    #             requires_human_input=requires_human,
    #         )
    #         if requires_human:
    #             break

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("vlm", self._vlm_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("state_router", self._state_router_node)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("analyze", self._analyze_node)
        # builder.add_node("backtrack", self._backtrack_node)

        builder.set_entry_point("vlm")
        builder.add_edge("vlm", "execute")
        builder.add_edge("execute", "state_router")
        builder.add_conditional_edges(
            "state_router",
            self._route_from_state,
            {
                "loop": "vlm",
                "analyze": "analyze",
                "human": "human_review",
                # "backtrack": "backtrack",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "analyze",
            self._route_from_state,
            {
                "loop": "vlm",
                # "backtrack": "backtrack",
                "end": END,
            },
        )
        # builder.add_conditional_edges(
        #     "backtrack",
        #     self._route_from_state,
        #     {
        #         "loop": "vlm",
        #         # "backtrack": "backtrack", # Continue backtracking if not reached
        #         "end": END,
        #     },
        # )
        builder.add_conditional_edges(
            "human_review",
            self._route_from_human_node,
            {
                "resume": "vlm",
                "abort": END,
            },
        )
        return builder.compile()

    def _vlm_node(self, state: AgentState) -> AgentState:
        screenshot = self._get_screen()
        
        thought_history_text = self._get_thought_history(state.get("history", []))
        full_instruction = VLM_GEMINI_USER_PROMPT.format(
            thought_history=thought_history_text,
            user_instruction=state["instruction"]
        )
        
        model_action = self.model_client.propose_action(full_instruction, screenshot.image)
        payload = model_action.payload
        
        return {
            "model_action": model_action,
            "payload": payload,
            "thought": payload.get("thought"),
            "status": "thinking",
            "pre_action_path": screenshot.path,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _execute_node(self, state: AgentState) -> AgentState:
        model_action = state.get("model_action")
        if model_action is None:
            raise RuntimeError("execute node received no model action")
            
        pre_screen = self._get_screen()
        adb_result = self.translator.execute(model_action.payload, img_size=pre_screen.image.size)
        post_screen = self._capture_screen()
        
        return {
            "status": "executed", # Factual: ADB command was sent
            "post_action_path": post_screen.path,
            "last_adb_commands": adb_result.commands,
        }

    def _state_router_node(self, state: AgentState) -> AgentState:
        """Observe results, update history, and determine next route."""
        post_screen_path = state.get("post_action_path")
        pre_screen_path = state.get("pre_action_path")
        
        new_screen_id = compute_screen_id(post_screen_path)
        
        # 1. Update Fact: Difference & Progress
        diff = compute_difference_from_paths(pre_screen_path, post_screen_path)
        progress = bool(diff is not None and diff >= self.progress_threshold)
        
        # 2. Update History
        iteration = state.get("iteration", 0)
        new_entry = {
            "iteration": iteration,
            "payload": state.get("payload", {}),
            "thought": state.get("thought"),
            "adb_commands": state.get("last_adb_commands", []),
            "progress": progress,
            "screen_id": new_screen_id,
        }
        history = list(state.get("history", [])) + [new_entry]
        
        # 3. Final factual status of this step
        status = "progress_made" if progress else "stagnant"
        if state.get("payload", {}).get("action") == "FINISH":
            status = "task_complete"
        elif state.get("payload", {}).get("interrupt"):
            status = "needs_human"

        # 4. Delegate Route Decision
        temp_state = cast(AgentState, {**state, "status": status, "history": history})
        route = self._determine_route(temp_state)

        return {
            "history": history,
            "status": status,
            "route": route,
            "progress": progress,
            "difference": diff,
            "current_screen_id": new_screen_id or state.get("current_screen_id"),
            "last_iteration_id": iteration
        }

    def _human_review_node(self, state: AgentState) -> AgentState:
        print("\n" + "="*50)
        print(" HUMAN ASSISTANCE REQUIRED ")
        print("="*50)
        breakpoint()
        payload = state.get("payload", {})
        last_thought = payload.get("interrupt").get("question", "에이전트가 정보를 더 필요로 합니다.")
        print(f"Agent: {last_thought}")
        
        user_input = input("\n질문에 대해 답변을 해주세요, 처음부터 다시하기를 원하시면 finish를 입력하세요: ").strip()
        
        if user_input.lower() == "finish":
            print(">>> 사용자의 요청으로 작업을 종료합니다.")
            return {
                "route": "abort",
                "status": "aborted",
            }
        
        # Update instruction by combining old instruction + agent's question + user's response
        combined_instruction = f"{state['instruction']}\n[추가 정보 Context]: {last_thought} -> {user_input}"
        
        print("\n>>> 정보를 업데이트했습니다. 다시 추론을 시작합니다...")
        return {
            "instruction": combined_instruction,
            "route": "loop",
            "status": "human_feedback_applied",
        }

    def _analyze_node(self, state: AgentState) -> AgentState:
        prompt = build_analysis_prompt(state)
        screen = self._get_screen(refresh=False)
        raw_analysis = self._invoke_analysis_model(prompt, screen.image)
        analysis_text, suggested_route, target_idx = parse_analysis_response(raw_analysis)
        
        # Decide next route
        if suggested_route == "backtrack":
            return {
                "analysis": analysis_text,
                "route": "backtrack",
                "status": "backtracking",
                "backtrack_target_index": target_idx
            }
            
        next_route = "loop" if suggested_route == "loop" and state.get("iteration", 0) < self.max_iterations else "end"
        status = "needs_retry" if next_route == "loop" else "analyzed"
        return {
            "analysis": analysis_text,
            "route": next_route,
            "status": status,
        }

    # def _backtrack_node(self, state: AgentState) -> AgentState:
    #     """Perform physical BACK actions to recover. Uses targeted replay if possible."""
    #     target_idx = state.get("backtrack_target_index")
    #     current_iter = state.get("iteration", 0)
        
    #     new_thought_history = list(state.get("thought_history", []))
        
    #     if target_idx is not None and target_idx < current_iter:
    #         num_backs = current_iter - target_idx
    #         print(f"Backtracking: Targeted index {target_idx} found. Executing {num_backs} BACK(s).")
    #         new_thought_history.append(f"System: Targeted backtrack to iteration {target_idx}. Executing {num_backs} BACKs.")
    #         for _ in range(num_backs):
    #             self.translator.execute({"action": "BACK"}, img_size=(1080, 1920))
    #     else:
    #         print("Backtracking: No specific target. Executing single ADB BACK.")
    #         new_thought_history.append("System: Performed single BACK to recover (Fallback).")
    #         self.translator.execute({"action": "BACK"}, img_size=(1080, 1920))
            
    #     post_screen = self._capture_screen()
    #     new_screen_id = compute_screen_id(post_screen.path)
        
    #     return {
    #         "status": "backtracked",
    #         "route": "loop",
    #         "thought_history": new_thought_history,
    #         "current_screen_id": new_screen_id,
    #         "last_iteration_id": target_idx if target_idx is not None else state.get("last_iteration_id", -1),
    #         "backtrack_target_index": None
    #     }

    def _route_from_state(self, state: AgentState) -> Literal["loop", "analyze", "human", "end", "backtrack"]:
        return state.get("route", "end")

    def _route_from_human_node(self, state: AgentState) -> Literal["resume", "abort"]:
        if state.get("route") == "abort" or state.get("status") == "aborted":
            return "abort"
        return "resume"

    def _get_thought_history(self, history: List[HistoryEntry]) -> str:
        """Derive thought history text from the history list."""
        items = []
        for entry in history:
            thought = entry.get("thought")
            if thought:
                items.append(f"Step {entry.get('iteration')}: {thought}")
        return "\n".join(items) or "Initial step."

    def _determine_route(self, state: AgentState) -> Literal["loop", "analyze", "human", "end"]:
        """Priority-based routing logic."""
        status = state.get("status", "")
        iteration = state.get("iteration", 0)
        history = state.get("history", [])

        # 1. End Condition
        if status in ["task_complete", "aborted"]:
            return "end"

        # 2. Human escalation: Explicit request or repeated stagnation
        stagnant_steps = sum(1 for h in history[-self.human_review_trigger:] if not h.get("progress", False))
        if status == "needs_human" or stagnant_steps >= self.human_review_trigger:
            return "human"

        # 3. Analysis: No progress or Max iterations reached
        if status == "stagnant" or iteration >= self.max_iterations:
            return "analyze"

        # 4. Default: Continue loop
        return "loop"

    def _invoke_analysis_model(self, prompt: str, image: Optional[Image.Image]) -> str:
        raw = self.analysis_client.generate(prompt, image)
        return self.model_client._coerce_raw_text(raw)

# Backwards compatibility for existing import sites
KioskAgent = LangGraphKioskAgent

if __name__ == "__main__":
    from kiosk_agent.src.config import AgentConfig, ADBConfig, ModelConfig, ScreenshotConfig
    
    # 더미 설정 생성
    dummy_config = AgentConfig(
        screenshot=ScreenshotConfig(adb_path="adb", device_id=None),
        model=ModelConfig(provider="gemini", gemini_api_key="dummy"),
        adb=ADBConfig(adb_path="adb")
    )
    
    # 에이전트 초기화 및 그래프 빌드
    agent = LangGraphKioskAgent(dummy_config)
    graph = agent._build_graph()
    
    # Mermaid PNG 생성 및 저장
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        output_path = "kiosk_agent_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {output_path}")
    except Exception as e:
        print(f"Could not generate graph image: {e}")
        print("Please ensure you have 'langgraph' and its dependencies installed correctly.")
