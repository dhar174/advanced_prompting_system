# feedback_manager.py

import json
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "conversation_logs.json")

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def log_conversation(conversation_history, feedback, questions):
    log_entry = {
        "conversation": conversation_history,
        "feedback": feedback,
        "questions": questions
    }
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(log_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
