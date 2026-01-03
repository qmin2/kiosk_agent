from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from kiosk_agent.src.models.base import ModelAction

class HistoryEntry(TypedDict, total=False):
    iteration: int
    payload: Dict[str, Any]
    thought: Optional[str]
    adb_commands: List[Sequence[str]]
    status: str
    pre_action_path: Optional[Path]
    post_action_path: Optional[Path]
    difference: Optional[float]
    progress: bool
    screen_id: Optional[str] # Visual fingerprint of the screen

class AgentState(TypedDict, total=False):
    instruction: str
    iteration: int
    model_action: Optional[ModelAction]
    raw_response: Optional[str]
    payload: Dict[str, Any]
    thought: Optional[str]
    adb_commands: List[Sequence[str]]
    status: str
    route: Literal["loop", "analyze", "human", "end", "backtrack"]
    pre_action_path: Optional[Path]
    post_action_path: Optional[Path]
    analysis: Optional[str]
    history: List[HistoryEntry]
    requires_human_input: bool
    human_decision: Optional[str]
    difference: Optional[float]
    progress: Optional[bool]
    last_adb_commands: List[Sequence[str]]
    
    # Advanced backtracking & structure tracking
    application_structure: str # Textual description of discovered UI map (formatted tree)
    ui_nodes: Dict[str, Any]    # screen_id -> { description: str, children: { action_desc: screen_id } }
    thought_tree: Dict[int, Any] # iteration_id -> { thought: str, parent: int, screen_id: str, action: str }
    thought_history: List[str] # Sequential history for prompting
    backtrack_target_index: Optional[int] # Index in history to return to
    last_iteration_id: int # The 'parent' iteration for the current step
    current_screen_id: Optional[str]

@dataclass
class AgentStepResult:
    screenshot_path: Optional[Path]
    raw_response: str
    thought: Optional[str]
    payload: Dict[str, Any]
    adb_commands: List[Sequence[str]]
    status: str
    analysis: Optional[str] = None
    history: List[HistoryEntry] | None = None

@dataclass
class AgentStreamEvent:
    stage: str
    state: AgentState
    message: str
    requires_human_input: bool = False
