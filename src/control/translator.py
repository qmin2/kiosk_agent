from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from kiosk_agent.src.config import ADBConfig
from kiosk_agent.src.control.adb import ADBController
from kiosk_agent.src.models.base import ModelAction


@dataclass
class ActionExecutionResult:
    commands: List[Sequence[str]]
    status: str


class ActionTranslator:
    """Converts structured model outputs into adb shell invocations."""

    def __init__(self, adb_controller: ADBController, config: ADBConfig):
        self.controller = adb_controller
        self.config = config

    def execute(
        self,
        model_action: Union[ModelAction, Mapping[str, Any], str],
        img_size: Tuple[int, int],
    ) -> ActionExecutionResult:
        """Translate the structured model output into adb commands."""

        payload = self._to_payload(model_action)
        action = payload.get("action")
        if not action:
            raise ValueError("Model response missing 'action'.")

        commands: List[Sequence[str]] = []
        status = payload.get("status", "completed")

        if action == "CLICK":
            point = self._extract_point(payload, img_size)
            commands.append(self.controller.tap(*point))
        elif action == "LONG_CLICK":
            point = self._extract_point(payload, img_size)
            commands.append(
                self.controller.run_raw(
                    [
                        "shell",
                        "input",
                        "touchscreen",
                        "swipe",
                        str(point[0]),
                        str(point[1]),
                        str(point[0]),
                        str(point[1]),
                        "500",
                    ]
                )
            )
        elif action == "SWIPE":
            start = self._extract_point(payload, img_size)
            end_hint = payload.get("end_position") or payload.get("target") or payload
            end = self._extract_point(end_hint, img_size)
            duration = payload.get("duration_ms", self.config.default_swipe_duration_ms)
            commands.append(self.controller.swipe(start[0], start[1], end[0], end[1], duration_ms=duration))
        elif action == "INPUT":
            value = payload.get("value")
            if not value:
                raise ValueError("INPUT action requires a non-empty 'value'.")
            commands.append(self.controller.type_text(value))
        elif action == "BACK":
            commands.append(self.controller.press("KEYCODE_BACK"))
        elif action == "HOME":
            commands.append(self.controller.press("KEYCODE_HOME"))
        elif action == "INTERRUPT":
            # INTERRUPT requires no ADB commands, but flags the need for human input
            status = "waiting_human"
        else:
            raise NotImplementedError(f"Unsupported action '{action}'.")

        return ActionExecutionResult(commands=commands, status=status)

    @staticmethod
    def _to_payload(model_action: Union[ModelAction, Mapping[str, Any], str]) -> Dict[str, Any]:
        if isinstance(model_action, ModelAction):
            return model_action.payload
        if isinstance(model_action, Mapping):
            return dict(model_action)
        if isinstance(model_action, str):
            return {"action": model_action}
        raise TypeError(f"Unsupported model_action type: {type(model_action)}")

    def _extract_point(self, model_action: Mapping[str, Any], img_size: Tuple[int, int]) -> Tuple[int, int]:
        point = model_action.get("position")
        if point is not None:
            if len(point) < 2:
                raise ValueError("Position must contain at least two values.")
            return self._normalized_position_to_pixels(point, img_size)

        box = model_action.get("box_2d")
        if box is not None:
            if len(box) != 4:
                raise ValueError("box_2d must contain four values.")
            top, left, bottom, right = self._normalized_box_to_pixels(box, img_size)
            x = int((left + right) / 2)
            y = int((top + bottom) / 2)
            return (x, y)

        raise ValueError("Model response missing either 'position' or 'box_2d'.")

    @staticmethod
    def _normalized_position_to_pixels(position: Sequence[float], img_size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = img_size
        x = int(max(0.0, min(1.0, position[0])) * width)
        y = int(max(0.0, min(1.0, position[1])) * height)
        return (x, y)

    @staticmethod
    def _normalized_box_to_pixels(box: Sequence[float], img_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        width, height = img_size
        top = int(max(0, min(1000, box[0])) / 1000 * height)
        left = int(max(0, min(1000, box[1])) / 1000 * width)
        bottom = int(max(0, min(1000, box[2])) / 1000 * height)
        right = int(max(0, min(1000, box[3])) / 1000 * width)
        return top, left, bottom, right
