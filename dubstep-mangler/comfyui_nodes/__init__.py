"""ComfyUI entry point. Symlink or copy this folder into ComfyUI/custom_nodes/.

If the dubstep_mangler engine isn't pip-installed into ComfyUI's Python, we
fall back to importing it from the repo layout next to this folder.
"""

import sys
from pathlib import Path

try:
    import dubstep_mangler  # noqa: F401
except ImportError:
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
