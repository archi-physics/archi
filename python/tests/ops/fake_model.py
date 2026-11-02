"""A fake OpenAI-compatible chat model for the ticket-triage lab (no network, no cost).

    python python/tests/ops/fake_model.py serve <port>

`POST /v1/chat/completions` answers a reply that cites the first `[node id]` of the user message (the ticket)
and, when the message lists one, the first related ticket, with token usage, as the drafter expects. Every
request is appended to `$FAKE_MODEL_LOG` (one JSON line: the model name and the user message's length) so the
lab can count the calls.
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CITED = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*:[^\]\s]+)\]")


def reply_for(user: str) -> str:
    ids = CITED.findall(user)
    ticket = ids[0] if ids else ""
    related = next((i for i in ids[1:] if i.startswith(ticket.split(":")[0] + ":")), "")
    text = f"Thank you for the report; the problem is described in your ticket [{ticket}]."
    if related:
        text += f" A similar ticket was resolved before [{related}]."
    return text + " An operator will follow up on it."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        user = next((m.get("content") or "" for m in body.get("messages") or [] if m.get("role") == "user"), "")
        if os.environ.get("FAKE_MODEL_LOG"):
            with open(os.environ["FAKE_MODEL_LOG"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": body.get("model"), "user_chars": len(user)}) + "\n")
        out = {"choices": [{"message": {"role": "assistant", "content": reply_for(user)}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 40}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    if sys.argv[1:2] != ["serve"]:
        sys.exit("usage: fake_model.py serve <port>")
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
