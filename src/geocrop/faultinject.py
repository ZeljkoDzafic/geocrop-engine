"""Fault-injection points for the reliability tests; inert unless explicitly enabled.

Long-running stages call `point(name)` once, in the middle of writing their outputs (`features` after the
first block of the stack, `train` after the model file but before its reports, `predict` after the first
block of the class/probability rasters). Nothing happens unless the environment variable
``GEOCROP_FAULT_PAUSE`` equals that name. Then the process

* writes ``"<name> <pid>"`` to the file named by ``GEOCROP_FAULT_SIGNAL`` (if set), so a test knows it arrived,
* sleeps ``GEOCROP_FAULT_SECONDS`` seconds (default 600),

so that `tests/test_reliability.py` can SIGKILL a real ``geocrop run`` with partial outputs on disk and check
that the rerun recovers. When the variable is unset the cost is one environment lookup per stage.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

PAUSE_ENV = "GEOCROP_FAULT_PAUSE"
SIGNAL_ENV = "GEOCROP_FAULT_SIGNAL"
SECONDS_ENV = "GEOCROP_FAULT_SECONDS"


def point(name: str) -> None:
    if os.environ.get(PAUSE_ENV) != name:
        return
    signal = os.environ.get(SIGNAL_ENV)
    if signal:
        tmp = Path(signal).with_name(Path(signal).name + ".tmp")
        tmp.write_text(f"{name} {os.getpid()}\n")
        os.replace(tmp, signal)
    time.sleep(float(os.environ.get(SECONDS_ENV, "600")))
