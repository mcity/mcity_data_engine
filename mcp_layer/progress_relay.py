"""Points chat_server.py's app-lifetime MCP log_handler at whichever
/chat/stream request is currently in flight. Single-flight only, same
assumption the existing run_auto_labeling streaming path already makes.
"""

_active_progress_cb = None


def set_active_progress_cb(cb):
    global _active_progress_cb
    _active_progress_cb = cb


def clear_active_progress_cb():
    global _active_progress_cb
    _active_progress_cb = None


def get_active_progress_cb():
    return _active_progress_cb
