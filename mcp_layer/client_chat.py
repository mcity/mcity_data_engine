import argparse
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

import requests
from dotenv import load_dotenv

from host_utils import resolve_host

load_dotenv()

host = resolve_host()
API_URL = f"http://{host}:8001/chat"
history = []


# Terminal mode

def send_message(message: str) -> str:
    global history
    payload = {"message": message, "history": history}
    res = requests.post(API_URL, json=payload)
    res.raise_for_status()
    reply = res.json().get("reply", "")
    history.append([message, reply])
    return reply


def run_terminal():
    print("MCity AI Agent is ready! Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        try:
            print(f"Agent: {send_message(user_input)}\n")
        except Exception as e:
            print(f"[error] {e}\n")


# UI server mode

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
UI_PORT = 5225


class _UIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def end_headers(self):
        # Prevent the browser from caching the UI so code changes take effect immediately.
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def _serve_ui():
    with TCPServer(("", UI_PORT), _UIHandler) as httpd:
        print(f"[UI] Serving {UI_DIR} at http://{host}:{UI_PORT}/")
        httpd.serve_forever()


def run_ui():
    if not os.path.isdir(UI_DIR):
        raise SystemExit(f"[UI] Missing folder: {UI_DIR}\nCreate it and put index.html inside.")
    t = threading.Thread(target=_serve_ui, daemon=True)
    t.start()
    time.sleep(0.4)
    url = f"http://{host}:{UI_PORT}/index.html"
    webbrowser.open(url)
    print(f"[UI] Opened {url}. Press Ctrl+C to stop.")
    t.join()



# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Launch web UI")
    args = parser.parse_args()
    run_ui() if args.ui else run_terminal()