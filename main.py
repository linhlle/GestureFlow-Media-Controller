"""GestureFlow entry point.

Kept as a thin shim so `python main.py` still works. The pipeline itself lives
in gestureflow/app.py, and the full command line is `python -m gestureflow`
(run, bench, record, replay, validate, bridge).
"""

from __future__ import annotations

import sys

from gestureflow.cli import main

if __name__ == "__main__":
    sys.exit(main(["run"] + sys.argv[1:]))
