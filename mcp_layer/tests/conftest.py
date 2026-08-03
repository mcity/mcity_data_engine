"""
Shared pytest configuration for mcp_layer unit tests.

Adds the project root and mcp_layer/ to sys.path so that:
  - config.config is importable (used by validate_workflow_state)
  - validate_workflow_state, chat_pipeline, etc. are importable with flat import names
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../mcity_data_engine_msight
_MCP_LAYER  = _REPO_ROOT / "mcp_layer"

for _p in (_REPO_ROOT, _MCP_LAYER):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_CONFIG_PATH = _REPO_ROOT / "config" / "config.py"


@pytest.fixture(autouse=True)
def _never_write_config_py(monkeypatch):
    """Stop any test from writing config/config.py.

    WorkflowState.save() rewrites the WORKFLOW_STATE line in the real repo file,
    and several MCP tools rewrite other parts of it. A test that reaches one of
    those paths silently dirties the working tree, so the write becomes a no-op
    for the whole suite. Tests that check persistence patch save() themselves.
    """
    real_write_text = Path.write_text

    def guarded(self, data, *args, **kwargs):
        if self.resolve() == _CONFIG_PATH:
            return len(data)
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded)
