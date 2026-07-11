"""Restricted worker for fixed, read-only agent tools.

This process intentionally has no generic command or code-execution operation.
It accepts one JSON request on stdin and emits one JSON response on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time

import psutil


def execute(tool: str, arguments: dict) -> dict:
    if tool == "system_status":
        battery = psutil.sensors_battery()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage(os.path.abspath(os.sep)).percent,
            "battery_percent": battery.percent if battery else None,
            "uptime_seconds": int(time.time() - psutil.boot_time()),
        }
    if tool == "disk_usage":
        usage = psutil.disk_usage(os.path.abspath(os.sep))
        return {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "percent": usage.percent,
        }
    if tool == "list_processes":
        limit = max(1, min(int(arguments.get("limit", 5)), 20))
        processes = []
        for process in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                processes.append(process.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        processes.sort(key=lambda item: (item.get("cpu_percent") or 0), reverse=True)
        return {"processes": processes[:limit]}
    raise ValueError("tool is not available in the restricted worker")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = execute(str(request.get("tool", "")), request.get("arguments", {}))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
