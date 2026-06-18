"""
Shared pytest configuration for mcp_layer unit tests.

Adds the project root and mcp_layer/ to sys.path so that:
  - config.config is importable (used by validate_workflow_state)
  - validate_workflow_state, chat_pipeline, etc. are importable with flat import names
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../mcity_data_engine_msight
_MCP_LAYER  = _REPO_ROOT / "mcp_layer"

for _p in (_REPO_ROOT, _MCP_LAYER):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
