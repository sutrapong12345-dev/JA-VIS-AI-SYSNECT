"""Structured tool definitions and validation for the J.A.R.V.I.S. agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List


ROLE_LEVEL = {"staff": 10, "manager": 20, "admin": 30}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    minimum_role: str
    risk: str
    approval_required: bool
    arguments: Dict[str, type]


TOOLS: Dict[str, ToolDefinition] = {
    "system_status": ToolDefinition(
        "system_status", "Read CPU, memory, disk and uptime", "staff", "low", False, {},
    ),
    "disk_usage": ToolDefinition(
        "disk_usage", "Read disk utilization without changing files", "staff", "low", False, {},
    ),
    "list_processes": ToolDefinition(
        "list_processes", "List the highest CPU processes", "manager", "medium", False,
        {"limit": int},
    ),
    "open_app": ToolDefinition(
        "open_app", "Open an allowlisted desktop application or HTTPS URL", "admin", "medium", True,
        {"target": str},
    ),
}


TOOL_TAG_RE = re.compile(r"\[TOOL:\s*(\{.*?\})\s*\]", re.IGNORECASE | re.DOTALL)


class ToolValidationError(ValueError):
    pass


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for raw in TOOL_TAG_RE.findall(text or ""):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolValidationError(f"invalid tool JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ToolValidationError("tool payload must be an object")
        calls.append(payload)
    return calls


def validate_tool_call(payload: Dict[str, Any], role: str) -> Dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    definition = TOOLS.get(name)
    if definition is None:
        raise ToolValidationError(f"unknown tool: {name}")
    if ROLE_LEVEL.get(role, 0) < ROLE_LEVEL[definition.minimum_role]:
        raise ToolValidationError(f"role {role} cannot use {name}")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolValidationError("arguments must be an object")
    unknown = set(arguments) - set(definition.arguments)
    if unknown:
        raise ToolValidationError(f"unknown arguments: {', '.join(sorted(unknown))}")
    normalized: Dict[str, Any] = {}
    for key, expected_type in definition.arguments.items():
        if key not in arguments:
            raise ToolValidationError(f"missing argument: {key}")
        value = arguments[key]
        if expected_type is int and isinstance(value, bool):
            raise ToolValidationError(f"{key} must be an integer")
        if not isinstance(value, expected_type):
            raise ToolValidationError(f"{key} must be {expected_type.__name__}")
        normalized[key] = value
    if name == "list_processes":
        normalized["limit"] = max(1, min(normalized["limit"], 20))
    if name == "open_app":
        normalized["target"] = normalized["target"].strip()[:500]
        if not normalized["target"]:
            raise ToolValidationError("target cannot be empty")
    return {
        "name": name,
        "arguments": normalized,
        "risk": definition.risk,
        "approval_required": definition.approval_required,
    }


def tool_catalog_for_prompt(role: str) -> str:
    available = [
        {
            "name": item.name,
            "description": item.description,
            "arguments": {key: value.__name__ for key, value in item.arguments.items()},
            "approval_required": item.approval_required,
        }
        for item in TOOLS.values()
        if ROLE_LEVEL.get(role, 0) >= ROLE_LEVEL[item.minimum_role]
    ]
    return json.dumps(available, ensure_ascii=False)
