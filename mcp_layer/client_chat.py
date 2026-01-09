# client_chat.py
# Usage:
#   python client_chat.py --ui    # web UI mode
#   python client_chat.py         # terminal mode (optional)

import argparse
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

import requests
from dotenv import load_dotenv


load_dotenv()
#host = os.getenv("PUBLIC_IP", "localhost")
import os
from dotenv import load_dotenv


load_dotenv()
host = os.getenv("PUBLIC_IP", "localhost")


def get_imds_token():
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}  # 6 hours
    try:
        response = requests.put(token_url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error getting token: {e}")
        return None


def get_metadata_with_token(path, token):
    url = f"http://169.254.169.254/latest/meta-data/{path}"
    headers = {"X-aws-ec2-metadata-token": token}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching metadata for {path}: {e}")
        return None

token = get_imds_token()
if token:
    host = get_metadata_with_token("public-ipv4", token)
else:
    host="localhost"
    print("Could not obtain IMDSv2 token.")

API_URL = f"http://{host}:8001/chat"
history = []

# -------- Terminal mode (optional) --------
def send_message(message: str):
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
            response = send_message(user_input)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"[error] {e}\n")

# -------- UI server mode --------
UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
UI_PORT = 5225

class _UIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

def _serve_ui():
    with TCPServer(("", UI_PORT), _UIHandler) as httpd:
        print(f"[UI] Serving {UI_DIR} at http://localhost:{UI_PORT}/")
        httpd.serve_forever()

def run_ui():
    if not os.path.isdir(UI_DIR):
        raise SystemExit(f"[UI] Missing folder: {UI_DIR}\nCreate it and put index.html inside.")
    t = threading.Thread(target=_serve_ui, daemon=True)
    t.start()
    time.sleep(0.4)
    url = f"http://localhost:{UI_PORT}/index.html"
    webbrowser.open(url)
    print(f"[UI] Opened {url}. Press Ctrl+C to stop.")
    t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Launch web UI (serves ./ui and opens browser)")
    args = parser.parse_args()

    if args.ui:
        run_ui()
    else:
        run_terminal()
