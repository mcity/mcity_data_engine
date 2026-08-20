#!/usr/bin/env bash
#
# Start the four components of the agentic interface in a tmux window,
# with one pane for each component. All four outputs stay visible.
#
#   Pane 0 (top left)      MCP Tool Server        port 8000
#   Pane 1 (top right)     Chat Server            port 8001
#   Pane 2 (bottom left)   Data Ingestion Server  port 8002
#   Pane 3 (bottom right)  Web UI                 port 5225
#
# Usage:
#   ./start_agent.sh          Start the components and attach to the panes.
#   ./start_agent.sh --kill   Stop the tmux session.
#
# tmux controls:
#   Ctrl+B then O            Go to the next pane.
#   Ctrl+B then Z            Make the selected pane full screen (again to undo).
#   Ctrl+B then [            Scroll in the selected pane (q to leave).
#   Ctrl+B then D            Leave the panes, but keep the components running.
#   Ctrl+C in a pane         Stop the component of that pane only.
#
# To come back after Ctrl+B D:  tmux attach -t mde_agent

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

SESSION="mde_agent"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

CMD_MCP='python mcp_layer/mcp_server.py'
CMD_CHAT='uvicorn mcp_layer.chat_server:app --port 8001 --reload --reload-exclude "config/config.py"'
CMD_INGEST='uvicorn mcp_layer.ingest_server:app --host 0.0.0.0 --port 8002 --reload'
CMD_UI='python mcp_layer/client_chat.py --ui'

case "${1:-}" in
    --kill) tmux kill-session -t "$SESSION" 2>/dev/null \
                && echo "Session $SESSION stopped." \
                || echo "No session $SESSION found."
            exit 0 ;;
    "")     ;;
    *)      echo "Unknown option: $1"; echo "Use --kill, or no option."; exit 1 ;;
esac

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Install it with: sudo apt install tmux"
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already runs. Attaching to it."
    echo "To start again: ./start_agent.sh --kill && ./start_agent.sh"
    exec tmux attach -t "$SESSION"
fi

# Make four empty panes first.
tmux new-session  -d -s "$SESSION" -n agent -c "$REPO_DIR"
tmux split-window -h -t "$SESSION:agent"   -c "$REPO_DIR"
tmux split-window -v -t "$SESSION:agent.0" -c "$REPO_DIR"
tmux split-window -v -t "$SESSION:agent.1" -c "$REPO_DIR"
tmux select-layout -t "$SESSION:agent" tiled

# Send the command to the shell of the pane. The shell stays alive after the
# command stops, so that the output of a component that stops stays visible.
run_in_pane() {
    local pane="$SESSION:agent.$1"
    local cmd="$2"
    if [ -f "$VENV_ACTIVATE" ]; then
        tmux send-keys -t "$pane" "source '$VENV_ACTIVATE'" C-m
    fi
    tmux send-keys -t "$pane" "$cmd" C-m
}

run_in_pane 0 "$CMD_MCP"
run_in_pane 1 "$CMD_CHAT"
run_in_pane 2 "$CMD_INGEST"
# The UI connects to the servers, so it waits for them to start.
run_in_pane 3 "sleep 8; $CMD_UI"

# Put a title on each pane border.
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_index}: #{pane_title} '
tmux select-pane -t "$SESSION:agent.0" -T 'MCP Tool Server :8000'
tmux select-pane -t "$SESSION:agent.1" -T 'Chat Server :8001'
tmux select-pane -t "$SESSION:agent.2" -T 'Ingest Server :8002'
tmux select-pane -t "$SESSION:agent.3" -T 'Web UI :5225'
tmux select-pane -t "$SESSION:agent.0"

echo "The web interface opens at http://localhost:5225"
exec tmux attach -t "$SESSION"
