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

from PIL import Image, ImageChops, ImageStat
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from kiosk_agent.src.config import AgentConfig, STTConfig
from kiosk_agent.src.control.adb import ADBController
from kiosk_agent.src.control.translator import ActionTranslator
from kiosk_agent.src.models.base import BaseModelClient, ModelAction
from kiosk_agent.src.models.chatgpt_client import ChatGPTClient
from kiosk_agent.src.models.gemini_client import GeminiClient
from kiosk_agent.src.models.local_vllm_client import LocalVLLMClient
from kiosk_agent.src.perception.screenshot import AndroidScreenshotter, ScreenshotResult
from kiosk_agent.src.prompts.prompts import ANALYSIS_PROMPT_TEMPLATE
from kiosk_agent.src.utils.utils import (
    parse_analysis_response, 
    compute_difference_from_paths,
    build_analysis_prompt,
    build_result,
    compute_screen_id,
    format_ui_tree,
    format_thought_history
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
        # Initialize checkpointer for saving state
        self.checkpointer = MemorySaver()
        # STT config
        self.stt_config = config.stt
        

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

    def _get_stt_input(self) -> tuple[Optional[str], str]:
        """Get user input via STT based on config settings.

        Returns:
            Tuple of (transcribed_text, input_source)
        """
        from kiosk_agent.src.stt import transcribe_from_file, transcribe_from_microphone, transcribe_streaming

        stt_config = self.stt_config

        if stt_config.mode == "file" and stt_config.file_path:
            input_source = f"stt_file ({stt_config.file_path})"
            text = transcribe_from_file(
                stt_config.file_path,
                language_code=stt_config.language_code
            )
        elif stt_config.mode == "streaming":
            input_source = "stt_streaming"
            text = transcribe_streaming(
                language_code=stt_config.language_code,
                sample_rate_hz=stt_config.sample_rate_hz
            )
        else:  # microphone
            input_source = f"stt_microphone ({stt_config.timeout_seconds}s)"
            text = transcribe_from_microphone(
                language_code=stt_config.language_code,
                sample_rate_hz=stt_config.sample_rate_hz,
                timeout_seconds=stt_config.timeout_seconds
            )

        return text, input_source

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
        
        # Set up configuration with thread_id (required for checkpointer)
        import uuid
        config = {
            "recursion_limit": 100,
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())}
        }

        # If we have a thread_id, we might be resuming.
        # If no previous_state is explicitly provided, we rely on the checkpointer.
        # However, invoke(initial_state) will overwrite if initial_state is not None.
        # If we want to resume, we should pass None as input state OR handle it carefully.
        # But here initial_state is always constructed.
        # LangGraph behavior: passing input state updates the state.

        # Strategy: If thread_id is used, we assume we might be resuming.
        # But if it's a NEW thread, we need initial state.
        # We can check if state exists for this thread.
        if thread_id:
            checkpoint = self.checkpointer.get(config)
            if checkpoint:
                # Resume: pass None or update dict?
                # passing None to invoke usually means "continue from last state"
                # But initial_state has 'instruction' etc.
                # If we want to resume we should probably only update new inputs if any.
                # For simplicity here, we pass initial_state to UPDATE the state,
                # but careful not to overwrite history if not intended.
                # Actually, simply calling invoke with state will MERGE/UPDATE.
                pass

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
            "adb_commands": [],
            "status": "init",
            "route": "loop",
            "history": [],
            "requires_human_input": False,
            "human_decision": None,
            "difference": None,
            "progress": None,
            "last_adb_commands": [],
            "application_structure": "",
            "ui_nodes": {},
            "thought_tree": {},
            "thought_history": [],
            "last_iteration_id": -1,
            "current_screen_id": None,
        }
        screen_id = compute_screen_id(self.current_screen_path)
        initial_state["current_screen_id"] = screen_id
        if screen_id:
            initial_state["ui_nodes"] = {screen_id: {"children": {}}}
            initial_state["application_structure"] = f"[Screen: {screen_id[:8]}]"
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
        builder.add_node("stt_input", self._stt_input_node)
        builder.add_node("vlm", self._vlm_node)
        builder.add_node("execute", self._execute_node)
        builder.add_node("state_router", self._state_router_node)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("analyze", self._analyze_node)
        builder.add_node("backtrack", self._backtrack_node)

        # Entry point depends on STT configuration
        if self.stt_config.enabled:
            builder.set_entry_point("stt_input")
            builder.add_conditional_edges(
                "stt_input",
                self._route_from_stt,
                {
                    "vlm": "vlm",
                    "end": END,
                },
            )
        else:
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
                "backtrack": "backtrack",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "analyze",
            self._route_from_state,
            {
                "loop": "vlm",
                "backtrack": "backtrack",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "backtrack",
            self._route_from_state,
            {
                "loop": "vlm",
                "backtrack": "backtrack", # Continue backtracking if not reached
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "human_review",
            self._route_from_human_node,
            {
                "resume": "vlm",
                "abort": END,
                "wait": END,
            },
        )
        return builder.compile(checkpointer=self.checkpointer)

    def _stt_input_node(self, state: AgentState) -> AgentState:
        """Get instruction via STT if enabled and no instruction provided."""
        instruction = state.get("instruction")
        instruction_source = state.get("instruction_source", "text")

        # If STT is enabled and no instruction yet, get it via STT
        if self.stt_config.enabled and not instruction:
            print("\n[STT] 음성 입력을 기다리는 중...")
            text, source = self._get_stt_input()

            if text:
                print(f"\n{'='*60}")
                print(f"[INPUT] 입력 방식: {source}")
                print(f"[INPUT] 인식된 명령: {text}")
                print(f"{'='*60}\n")
                return {
                    "instruction": text,
                    "instruction_source": source,
                    "status": "instruction_received",
                    "route": "vlm",
                }
            else:
                print("[STT] 음성이 감지되지 않았습니다.")
                return {
                    "status": "no_instruction",
                    "route": "end",
                }

        # Instruction already provided (text input)
        if instruction:
            print(f"\n{'='*60}")
            print(f"[INPUT] 입력 방식: {instruction_source}")
            print(f"[INPUT] 명령: {instruction}")
            print(f"{'='*60}\n")
            return {
                "status": "instruction_received",
                "route": "vlm",
            }

        # No instruction and STT not enabled
        print("[ERROR] instruction이 필요합니다.")
        return {
            "status": "no_instruction",
            "route": "end",
        }

    def _route_from_stt(self, state: AgentState) -> Literal["vlm", "end"]:
        """Route based on STT input result."""
        route = state.get("route", "end")
        if route == "vlm":
            return "vlm"
        return "end"

    def _vlm_node(self, state: AgentState) -> AgentState:
        screenshot = self._get_screen()
        
        # Build contextual instruction
        from kiosk_agent.src.prompts.vlm_system_prompt import VLM_GEMINI_USER_PROMPT
        app_structure = state.get("application_structure") or "Discovered menu items and screens will be listed here."
        
        # Flatten tree to sequential history for LLM
        current_path_history = format_thought_history(state.get("thought_tree", {}), state.get("iteration", 0))
        thought_history_text = "\n".join(current_path_history) or "Initial step."
        
        full_instruction = VLM_GEMINI_USER_PROMPT.format(
            application_structure=app_structure,
            thought_history=thought_history_text,
            user_instruction=state["instruction"]
        )

        model_action = self.model_client.propose_action(full_instruction, screenshot.image)
        payload = model_action.payload
        iteration = state.get("iteration", 0) + 1
        
        return {
            "model_action": model_action,
            "raw_response": model_action.raw_text,
            "payload": payload,
            "thought": payload.get("thought"),
            "status": "action_proposed",
            "pre_action_path": screenshot.path,
            "iteration": iteration,
        }

    def _execute_node(self, state: AgentState) -> AgentState:
        model_action = state.get("model_action")
        if model_action is None:
            raise RuntimeError("execute node received no model action")
        pre_screen = self._get_screen()
        adb_result = self.translator.execute(model_action.payload, img_size=pre_screen.image.size)
        post_screen = self._capture_screen()
        accumulated_commands = list(state.get("adb_commands", [])) + adb_result.commands
        return {
            "adb_commands": accumulated_commands,
            "status": adb_result.status,
            "post_action_path": post_screen.path,
            "last_adb_commands": adb_result.commands,
        }

    def _state_router_node(self, state: AgentState) -> AgentState:
        
        post_screen_path = state.get("post_action_path")
        new_screen_id = compute_screen_id(post_screen_path)
        old_screen_id = state.get("current_screen_id")
        
        difference = compute_difference_from_paths(
            state.get("pre_action_path"), post_screen_path
        )
        
        # Progress check
        progress = bool(difference is not None and difference >= self.progress_threshold)
        
        # Check if it was just a scroll
        is_scroll = state.get("payload", {}).get("action") in {"SWIPE", "SCROLL"}
        screen_changed = new_screen_id != old_screen_id
        
        # Update Thought Tree and Structure
        thought_tree = dict(state.get("thought_tree", {}))
        last_id = state.get("last_iteration_id", -1)
        current_iter = state.get("iteration", 0)
        thought = state.get("thought")
        
        if thought:
            thought_tree[current_iter] = {
                "thought": thought,
                "parent": last_id,
                "screen_id": old_screen_id  # Thought happened at the pre-action screen
            }
        
        # Sequentially formatted history for current logs and AgentState tracking
        thought_history = format_thought_history(thought_tree, current_iter)
        
        # Update Tree Structure
        ui_nodes = dict(state.get("ui_nodes", {}))
        if old_screen_id and new_screen_id and screen_changed and not is_scroll and progress:
            # Add edge to the tree
            action_desc = f"{state.get('payload', {}).get('action')} at {state.get('payload', {}).get('box_2d')}"
            
            # Ensure old node exists
            if old_screen_id not in ui_nodes:
                ui_nodes[old_screen_id] = {"children": {}}
                
            # Update parent
            ui_nodes[old_screen_id]["children"][action_desc] = new_screen_id
            
            # Initialize new node
            if new_screen_id not in ui_nodes:
                ui_nodes[new_screen_id] = {"children": {}}

        # Format the tree for the prompt
        root_id = list(ui_nodes.keys())[0] if ui_nodes else new_screen_id
        formatted_structure = format_ui_tree(ui_nodes, root_id) if root_id else ""

        history = list(state.get("history", []))
        history.append(
            {
                "iteration": state.get("iteration", 0),
                "payload": state.get("payload", {}),
                "thought": state.get("thought"),
                "adb_commands": state.get("last_adb_commands", []),
                "status": state.get("status", "unknown"),
                "pre_action_path": state.get("pre_action_path"),
                "post_action_path": state.get("post_action_path"),
                "difference": difference,
                "progress": progress,
                "screen_id": new_screen_id,
            }
        )

        enriched_state = dict(state)
        enriched_state["history"] = history
        enriched_state["difference"] = difference
        enriched_state["progress"] = progress
        enriched_state["status"] = "continue" if progress else "no_progress"
        enriched_state["current_screen_id"] = new_screen_id or old_screen_id
        
        route = self._determine_route(enriched_state)

        return {
            "route": route,
            "history": history,
            "difference": difference,
            "progress": progress,
            "thought_history": thought_history,
            "ui_nodes": ui_nodes,
            "thought_tree": thought_tree,
            "application_structure": formatted_structure,
            "current_screen_id": new_screen_id or old_screen_id,
            "last_iteration_id": current_iter,
            "requires_human_input": route == "human",
            "status": "waiting_human" if route == "human" else enriched_state["status"]
        }

    def _human_review_node(self, state: AgentState) -> AgentState:
        """Handle human review / INTERRUPT action.

        If STT is enabled, prompts user for voice input.
        Otherwise, checks for human_decision in state.
        """
        # Get interrupt info from payload if available
        payload = state.get("payload", {})
        interrupt_info = payload.get("interrupt", {})
        question = interrupt_info.get("question", "추가 입력이 필요합니다. 말씀해 주세요.")
        reason = interrupt_info.get("reason", "HUMAN_INPUT_REQUIRED")

        # If STT is enabled, get voice input
        if self.stt_config.enabled:
            print(f"\n{'='*60}")
            print(f"[INTERRUPT] 이유: {reason}")
            print(f"[INTERRUPT] 질문: {question}")
            print(f"{'='*60}")
            print("[STT] 음성 입력을 기다리는 중...")

            text, _ = self._get_stt_input()

            if text:
                print(f"[STT] 사용자 응답: {text}")
                # Update instruction with user's response and continue
                current_instruction = state.get("instruction", "")
                updated_instruction = f"{current_instruction} (사용자 추가 입력: {text})"
                return {
                    "instruction": updated_instruction,
                    "human_decision": text,
                    "route": "loop",
                    "requires_human_input": False,
                    "status": "human_feedback_applied",
                }
            else:
                print("[STT] 음성이 감지되지 않았습니다. 다시 시도해 주세요.")
                return {
                    "status": "waiting_human",
                    "route": "wait",
                    "requires_human_input": True,
                }

        # Fallback: check for human_decision in state (non-STT mode)
        decision = (state.get("human_decision") or "").lower()
        if not decision:
            print(f"\n[INTERRUPT] 이유: {reason}")
            print(f"[INTERRUPT] 질문: {question}")
            print("[WAITING] human_decision 입력을 기다리는 중...")
            return {
                "status": "waiting_human",
                "route": "wait",
                "requires_human_input": True,
            }
        if decision == "resume":
            return {
                "route": "loop",
                "requires_human_input": False,
                "status": "human_feedback_applied",
            }
        return {
            "route": "abort",
            "requires_human_input": False,
            "status": "aborted",
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

    def _backtrack_node(self, state: AgentState) -> AgentState:
        """Perform physical BACK actions to recover. Uses targeted replay if possible."""
        target_idx = state.get("backtrack_target_index")
        history = state.get("history") or []
        current_iter = state.get("iteration", 0)
        
        thought_history = list(state.get("thought_history", []))
        
        if target_idx is not None and target_idx < current_iter:
            num_backs = current_iter - target_idx
            print(f"Backtracking: Targeted index {target_idx} found. Executing {num_backs} BACK(s).")
            thought_history.append(f"System: Targeted backtrack to iteration {target_idx}. Executing {num_backs} BACKs.")
            for _ in range(num_backs):
                self.translator.execute({"action": "BACK"}, img_size=(1080, 1920))
        else:
            print("Backtracking: No specific target. Executing single ADB BACK.")
            thought_history.append("System: Performed single BACK to recover (Fallback).")
            self.translator.execute({"action": "BACK"}, img_size=(1080, 1920))
            
        post_screen = self._capture_screen()
        new_screen_id = compute_screen_id(post_screen.path)
        
        return {
            "status": "backtracked",
            "route": "loop",
            "thought_history": thought_history,
            "current_screen_id": new_screen_id,
            "last_iteration_id": target_idx if target_idx is not None else state.get("last_iteration_id", -1),
            "backtrack_target_index": None # Reset
        }

    def _route_from_state(self, state: AgentState) -> Literal["loop", "analyze", "human", "end", "backtrack"]:
        return state.get("route", "end")

    def _route_from_human_node(self, state: AgentState) -> Literal["resume", "abort", "wait"]:
        if state.get("route") == "abort" or state.get("status") == "aborted":
            return "abort"
        if state.get("route") == "loop" or state.get("status") == "human_feedback_applied":
            return "resume"
        return "wait"

    def _determine_route(self, state: AgentState) -> Literal["loop", "analyze", "human", "end", "backtrack"]:
        status = (state.get("status") or "").lower()
        iteration = state.get("iteration", 0)
        history = state.get("history") or []
        last_step = history[-1] if history else None
        progress = bool(last_step and last_step.get("progress"))

        # Check if last action was SWIPE/SCROLL - be more lenient with progress check
        last_action = state.get("payload", {}).get("action", "").upper()
        is_scroll_action = last_action in {"SWIPE", "SCROLL"}

        if self._should_request_human_input(history, status):
            return "human"

        if status in {"needs_analysis", "analyze"}:
            return "analyze"

        if status == "backtracking":
            return "backtrack"

        # For SWIPE/SCROLL actions, continue looping even without significant progress
        if is_scroll_action and iteration < self.max_iterations:
            print(f"[DEBUG] Route: loop (SWIPE/SCROLL action, iteration={iteration})")
            return "loop"

        if not progress:
            # If no progress for N steps, analyze (which might trigger backtrack)
            print(f"[DEBUG] Route: analyze (no progress, iteration={iteration})")
            return "analyze"

        if status in {"retry", "continue", "needs_retry"} and iteration < self.max_iterations:
            print(f"[DEBUG] Route: loop (status={status}, iteration={iteration})")
            return "loop"

        if iteration >= self.max_iterations:
            print(f"[DEBUG] Route: analyze (max_iterations reached, iteration={iteration})")
            return "analyze"

        if status in {"completed", "success", "analyzed"}:
            print(f"[DEBUG] Route: end (status={status})")
            return "end"

        print(f"[DEBUG] Route: end (fallback, status={status}, iteration={iteration}, progress={progress})")
        return "end"


    def _invoke_analysis_model(self, prompt: str, image: Optional[Image.Image]) -> str:
        raw = self.analysis_client.generate(prompt, image)
        return self.model_client._coerce_raw_text(raw)

    def _should_request_human_input(self, history: List[HistoryEntry], status: str) -> bool:
        status = status.lower()
        if status in {"needs_human", "awaiting_human", "waiting_human"}:
            return True
        if not history:
            return False
        latest_window = history[-self.human_review_trigger :]
        if len(latest_window) < self.human_review_trigger:
            return False
        return all(not entry.get("progress") for entry in latest_window)

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
