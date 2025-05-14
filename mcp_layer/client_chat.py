# mcp_layer/client.py
import requests

API_URL = "http://localhost:8001/chat"
history = []

def send_message(message: str):
    global history
    payload = {"message": message, "history": history}
    res = requests.post(API_URL, json=payload)
    reply = res.json()["reply"]
    history.append([message, reply])
    return reply

def run_chat():
    print("💬 MCity AutoLabeling Agent is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        response = send_message(user_input)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    run_chat()
