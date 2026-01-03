from kiosk_agent.src.models.base import BaseModelClient, ModelAction
from kiosk_agent.src.utils.types import AgentState, AgentStepResult
from kiosk_agent.src.prompts.prompts import ANALYSIS_PROMPT_TEMPLATE
import json

from typing import Literal, Optional
from pathlib import Path
from PIL import Image, ImageChops, ImageStat


def parse_analysis_response(raw: str) -> tuple[str, Literal["loop", "end", "backtrack"], Optional[int]]:
    try:
        payload = BaseModelClient._safe_parse(raw)
    except ValueError:
        return raw, "end", None
        
    thought = payload.get("thought") or payload.get("analysis") or raw
    next_hint = (payload.get("next") or payload.get("route") or "end").lower()
    target_idx = payload.get("target_step_index")
    
    if next_hint in {"vlm", "loop", "retry"}:
        return thought, "loop", None
    if next_hint in {"backtrack", "back"}:
        return thought, "backtrack", target_idx
    return thought, "end", None

def compute_difference_from_paths(pre_path: Optional[Path], post_path: Optional[Path]) -> Optional[float]:
    if not pre_path or not post_path:
        return None
    try:
        with Image.open(pre_path) as before_img, Image.open(post_path) as after_img:
            before = before_img.convert("RGB")
            after = after_img.convert("RGB")
    except (FileNotFoundError, OSError):
        return None
    
    if before.size != after.size:
        after = after.resize(before.size)
    diff = ImageChops.difference(before, after)
    stat = ImageStat.Stat(diff)
    if not stat.mean:
        return None
    mean = sum(stat.mean) / len(stat.mean)

    return mean / 255

def compute_screen_id(image_path: Optional[Path]) -> Optional[str]:
    """Compute a simple visual fingerprint for the screen."""
    if not image_path or not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            # Resize to small and convert to grayscale for a fuzzy hash
            small = img.resize((16, 16)).convert("L")
            pixels = list(small.getdata())
            # Convert to a hex-like string
            return "".join(f"{p:02x}" for p in pixels)
    except Exception:
        return None

def build_analysis_prompt(state: AgentState) -> str:
    payload = state.get("payload") or {}
    adb_commands = state.get("adb_commands", [])
    commands_text = "\n".join(" ".join(cmd) for cmd in adb_commands) or "(no commands executed)"
    app_structure = state.get("application_structure") or "Not yet defined."
    thought_history = "\n".join(state.get("thought_history") or []) or "No previous thoughts."
    
    try:
        serialized_payload = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        serialized_payload = str(payload)
        
    return ANALYSIS_PROMPT_TEMPLATE.format(
        instruction=state.get("instruction"),
        serialized_payload=serialized_payload,
        commands_text=commands_text,
        application_structure=app_structure,
        thought_history=thought_history
    )
    
def build_result(state: AgentState) -> AgentStepResult:
    payload = state.get("payload") or {}
    history = state.get("history") or []
    last_screenshot = state.get("post_action_path")
    if history:
        last_entry = history[-1]
        last_screenshot = last_entry.get("post_action_path") or last_screenshot
    return AgentStepResult(
        screenshot_path=last_screenshot,
        raw_response=state.get("raw_response") or "",
        thought=state.get("thought"),
        payload=payload,
        adb_commands=state.get("adb_commands", []),
        status=state.get("status", "unknown"),
        analysis=state.get("analysis"),
        history=history,
    )

def format_ui_tree(ui_nodes: Dict[str, Any], current_node_id: str, visited=None, depth=0) -> str:
    """Format the UI relationship graph as a pretty tree string."""
    if visited is None:
        visited = set()
    
    if current_node_id not in ui_nodes or current_node_id in visited:
        return ""
    
    visited.add(current_node_id)
    node = ui_nodes[current_node_id]
    short_id = current_node_id[:8]
    indent = "  " * depth
    branch = "└── " if depth > 0 else ""
    
    output = f"{indent}{branch}[Screen: {short_id}]\n"
    
    children = node.get("children", {})
    for action_desc, child_id in children.items():
        output += f"{indent}    ({action_desc})\n"
        output += format_ui_tree(ui_nodes, child_id, visited, depth + 1)
        
    return output

def format_thought_history(thought_tree: Dict[int, Any], current_iter: int) -> List[str]:
    """
    Generate a sequential thought history from a tree structure.
    Currently follows a chronological order of the active branch/exploration.
    """
    history = []
    # Sort keys to ensure chronological consistency in the log
    for iter_id in sorted(thought_tree.keys()):
        node = thought_tree[iter_id]
        sid = (node.get("screen_id") or "unknown")[:8]
        thought = node.get("thought", "")
        # action = node.get("action", "") # Usually included in thought or seen in ADB commands
        if thought:
            history.append(f"Step {iter_id} [Screen: {sid}]: {thought}")
    return history
