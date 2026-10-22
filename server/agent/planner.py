"""LLM planner (F6.3): Bedrock is asked what to do only at decision points, never per frame.

The state machine emits {"type": "planner", "event": ...} when a fixed rule isn't enough
(no candidate for 20 s, bad frames for 8 s, not found). We send Claude a compact JSON observation
and force it to answer with one or two tool calls, which become phone actions.
Any Bedrock error or timeout -> no actions (the state machine already spoke a fixed phrase).
"""
import json
import os
import time

import boto3
from botocore.config import Config

MODEL = os.getenv("FINDMINE_PLANNER_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")

SYSTEM = """You are the planner inside FindMine, a phone app that helps a blind or low-vision person find \
their own belongings. The phone camera is analysed by OpenCV; you receive a short JSON summary when the \
search needs a decision. Choose the next step with your tools.
Rules:
- Speak in the user's language (field "lang": en, fr or ar). One short, concrete sentence per speak call.
- Never say the item was found or that you can see it: only the vision system decides that.
- Prefer physical, doable actions: turn the torch on in the dark, sweep slowly, point lower at the floor or \
table, check a likely place (use last_seen if given).
- If the picture keeps being bad, explain the fix differently from the coach's last message.
- Use at most two tools."""

TOOLS = [
    {"toolSpec": {"name": "speak", "description": "Say one short sentence to the user.",
                  "inputSchema": {"json": {"type": "object", "required": ["text"],
                                           "properties": {"text": {"type": "string"}}}}}},
    {"toolSpec": {"name": "set_torch", "description": "Turn the phone torch on or off (helps in the dark).",
                  "inputSchema": {"json": {"type": "object", "required": ["on"],
                                           "properties": {"on": {"type": "boolean"}}}}}},
    {"toolSpec": {"name": "ask_user", "description": "Ask the user a question with a few short answer options.",
                  "inputSchema": {"json": {"type": "object", "required": ["question", "options"],
                                           "properties": {"question": {"type": "string"},
                                                          "options": {"type": "array", "items": {"type": "string"}}}}}}},
]

_client = None


def client():
    global _client
    if _client is None:  # created on first use so the server starts without AWS keys
        _client = boto3.client("bedrock-runtime", region_name=REGION,
                               config=Config(read_timeout=6, connect_timeout=3, retries={"max_attempts": 1}))
    return _client


def to_actions(tool, args, lang):
    if tool == "speak":
        return [{"type": "speak", "key": "planner", "text": {lang: args["text"]}}]
    if tool == "set_torch":
        return [{"type": "torch", "on": bool(args["on"])}]
    if tool == "ask_user":
        return [{"type": "speak", "key": "planner", "text": {lang: args["question"]}},
                {"type": "ask_user", "options": list(args["options"])[:4]}]
    return []


def plan(observation, bedrock=None):
    """observation: dict with at least "event" and "lang". -> (actions, trace record)."""
    t = time.perf_counter()
    rec = {"planner_event": observation.get("event"), "model": MODEL}
    try:
        resp = (bedrock or client()).converse(
            modelId=MODEL, system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": json.dumps(observation, ensure_ascii=False)}]}],
            toolConfig={"tools": TOOLS, "toolChoice": {"any": {}}},
            inferenceConfig={"maxTokens": 300, "temperature": 0})
        calls = [c["toolUse"] for c in resp["output"]["message"]["content"] if "toolUse" in c][:2]
        acts = [a for c in calls for a in to_actions(c["name"], c["input"], observation.get("lang", "en"))]
        rec["tools"] = [{"name": c["name"], "input": c["input"]} for c in calls]
    except Exception as e:  # noqa: BLE001 - planner must never break the search
        acts, rec["error"] = [], f"{type(e).__name__}: {e}"[:200]
    rec["latency_ms"] = round(1000 * (time.perf_counter() - t))
    return acts, rec
