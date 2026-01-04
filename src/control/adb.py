from __future__ import annotations

import shlex
import subprocess
from typing import List, Sequence

from kiosk_agent.src.config import ADBConfig
import time

class ADBController:
    """Light wrapper around adb shell gestures."""

    def __init__(self, config: ADBConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def _base_cmd(self) -> List[str]:
        # cmd = [self.config.adb_path]
        # if self.config.device_id:
        #     cmd += ["-s", self.config.device_id]
        cmd = [self.config.adb_path]
        return cmd

    def tap(self, x: int, y: int) -> Sequence[str]:
        # tap(=click) (y,x) position.
        return self._run(["shell", "input", "touchscreen", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms=100) -> Sequence[str]:
        cmd = ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)]
        return self._run(cmd)

    def press(self, key: str) -> Sequence[str]:
        return self._run(["shell", "input", "keyevent", key])

    def type_text(self, text: str) -> Sequence[str]:
        safe_text = text.replace(" ", "%s")
        return self._run(["shell", "input", "text", safe_text])

    def run_raw(self, args: Sequence[str]) -> Sequence[str]:
        return self._run(list(args))
    
    def screenshot(self):
        abs_path = self.config.screenshot_abs_path
        new_path = abs_path +"/" + "mac_test_" + str(self.config.steps) + ".png"
        self.config.steps += 1

        return self._run(["exec-out", "screencap", "-p", ">", f"{new_path}"])

    def _run(self, args: Sequence[str]) -> Sequence[str]:
        full_cmd = self._base_cmd() + list(args)
        # if self.dry_run:
        #     print(" ".join(shlex.quote(token) for token in full_cmd))
        #     return full_cmd

        # final_cmd = " ".join(shlex.quote(token) for token in full_cmd)
        final_cmd = " ".join(full_cmd)
        subprocess.run(final_cmd, shell= True, check=True)
        time.sleep(1.5) # 동작완료 대기
        return full_cmd

