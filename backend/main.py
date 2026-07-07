"""
J.A.R.V.I.S. — AI Agent Backend
================================
FastAPI backend for the J.A.R.V.I.S. local AI assistant.

Design goals for the move toward an Enterprise AI Agent Platform:
  * Provider abstraction (Anthropic / Gemini / Ollama) behind one interface
  * Per-session memory (no shared global state) so multiple clients are isolated
  * Config driven entirely by environment variables (no secrets in code)
  * Safe-by-default command execution (opt-in, allowlisted)
  * Structured logging and typed error handling

Run:  python backend/main.py
"""

from __future__ import annotations

import base64
import glob
import hmac
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import subprocess
import sys
import urllib.request
import urllib.parse
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #
# Load .env from next to this file so config is found regardless of the
# working directory the server is launched from.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")        # EVERY action gets logged here (jarvis.log) + per-session chat JSON
TRAINING_DIR = os.path.join(ROOT_DIR, "training")  # fine-tune datasets are built here
os.makedirs(LOGS_DIR, exist_ok=True)

# Every action JARVIS takes (chat, commands, admin logins, vision/clipboard,
# searches, knowledge changes, errors...) is logged to logs/jarvis.log, rotated
# at 5MB x 5 backups so it never grows unbounded, in addition to the console.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(LOGS_DIR, "jarvis.log"), maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("jarvis")

# Ring buffer of the most recent log lines — a fast in-memory tail for the
# frontend's SYSTEM LOG panel (/api/system-log). The permanent, complete history
# lives in logs/jarvis.log; this is just a quick recent-activity cache.
_log_buffer: deque = deque(maxlen=50)


class _RingBufferLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append({
                "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": record.getMessage(),
            })
        except Exception:  # noqa: BLE001 - logging must never crash the app
            pass


log.addHandler(_RingBufferLogHandler())

IS_WINDOWS = sys.platform.startswith("win")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_session_key(session_id: str) -> str:
    """Sanitize a client-supplied session_id so it can NEVER be used to escape a
    directory when interpolated into a filename (e.g. '../../evil', absolute
    paths, NUL bytes). Keeps only [A-Za-z0-9_.-], caps length, strips leading
    dots so it can't become a hidden/relative path. Path-traversal hardening."""
    cleaned = _SAFE_ID_RE.sub("_", (session_id or "").strip())[:128].lstrip(".")
    return cleaned or "default"


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Central configuration loaded from the environment (never hard-coded secrets)."""

    active_ai: str = os.getenv("ACTIVE_AI", "gemini").strip().lower()

    # Branding / company identity
    company_name: str = os.getenv("COMPANY_NAME", "SYSNECT")
    assistant_name: str = os.getenv("ASSISTANT_NAME", "J.A.R.V.I.S.")

    # Company knowledge base
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIR)
    knowledge_char_limit: int = int(os.getenv("KNOWLEDGE_CHAR_LIMIT", "12000"))

    # SYSNECT internal data search — real project folders on this machine that
    # the admin can full-text search on demand (separate from the always-injected
    # knowledge base above, so there is no fixed char-limit ceiling on coverage).
    sysnect_data_dirs: List[str] = field(default_factory=lambda: _env_list(
        "SYSNECT_DATA_DIRS",
        ",".join([
            r"D:\AI Data\sysnect-local-ai",
            r"D:\SYSNECT  WORK SPACE\Sysnect Project Ticket\Sysnect html",
            ROOT_DIR,
        ]),
    ))

    # Admin login (Authentication modal — real values live in backend/.env only)
    admin_username: str = os.getenv("ADMIN_USERNAME", "sutrapongadmin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")  # empty = login disabled

    # Superadmin security override code — a SEPARATE secret (not the login
    # password) that can lift a security lockdown even without an existing
    # authenticated session. Deliberately decoupled from admin_password: login
    # itself is blocked during lockdown by design, so this is the break-glass
    # path. Empty = feature disabled (lockdown can only be cleared by a session
    # that was already admin before the lockdown engaged).
    security_override_code: str = os.getenv("SECURITY_OVERRIDE_CODE", "")

    # Anthropic / Claude
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    # Kept at the original model to preserve existing behavior. Note: this ID is
    # retired on the official Anthropic API (since Apr 2026) — if Claude returns a
    # 404, set CLAUDE_MODEL=claude-haiku-4-5, or point CLAUDE_BASE_URL at a gateway.
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    claude_base_url: str = os.getenv("CLAUDE_BASE_URL", "")  # optional gateway (e.g. Z.ai)

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Ollama (OpenAI-compatible)
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")

    # Groq (OpenAI-compatible cloud — very fast inference)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Generation
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
    max_history: int = int(os.getenv("MAX_HISTORY", "20"))
    doc_context_limit: int = int(os.getenv("DOC_CONTEXT_LIMIT", "15000"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

    # Security
    # Defaults preserve the original behavior (commands enabled, open anything).
    # To lock down: set ENABLE_COMMAND_EXECUTION=false, or ENABLE_SHELL_COMMANDS=false
    # to allow only OPEN, or set COMMAND_ALLOWLIST to restrict which apps OPEN may launch.
    cors_origins: List[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))
    enable_command_execution: bool = _env_bool("ENABLE_COMMAND_EXECUTION", True)
    enable_shell_commands: bool = _env_bool("ENABLE_SHELL_COMMANDS", True)
    command_allowlist: List[str] = field(
        default_factory=lambda: _env_list("COMMAND_ALLOWLIST", "")  # empty = allow all (original)
    )
    # Last-resort guard: even in GOD MODE, block a handful of IRREVERSIBLE,
    # machine-destroying commands (format/wipe a whole drive, diskpart clean).
    # This protects the boss's OWN PC from a hallucinating model or an injected
    # command (web search / uploaded docs are injection vectors). Everything else
    # — open apps, run normal PowerShell — still works. Set to true to remove it.
    allow_destructive_commands: bool = _env_bool("ALLOW_DESTRUCTIVE_COMMANDS", False)


settings = Settings()


# --------------------------------------------------------------------------- #
#  Errors
# --------------------------------------------------------------------------- #
class ProviderNotConfigured(Exception):
    def __init__(self, env_name: str):
        self.env_name = env_name
        super().__init__(f"{env_name} is not set")


# --------------------------------------------------------------------------- #
#  Session store (per-session memory — replaces the old global chat_history)
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    history: List[Dict[str, str]] = field(default_factory=list)
    document: str = ""
    is_admin: bool = False
    last_seen: float = field(default_factory=time.time)
    # A destructive action (e.g. a delete-ish shell command) waiting for the
    # admin password to be re-entered before it actually runs.
    pending_action: Optional[Dict[str, object]] = None


class SessionStore:
    """Thread-safe in-memory session store.

    For a production/multi-node deployment this should be swapped for Redis or a
    database; the interface is kept small so that swap is straightforward.
    """

    def __init__(self, max_history: int, ttl_seconds: int):
        self._max_history = max_history
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session()
            skey = safe_session_key(session_id)
            # Try to load history from disk if it exists
            log_path = os.path.join(LOGS_DIR, f"chat_log_{skey}.json")
            if os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as fh:
                        history = json.load(fh)
                        if isinstance(history, list):
                            session.history = history
                except Exception as exc:
                    log.error("Failed to load session history from disk for %s: %s", session_id, exc)
            
            # Try to load admin metadata from disk if it exists
            meta_path = os.path.join(LOGS_DIR, f"session_metadata_{skey}.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                        if isinstance(meta, dict):
                            session.is_admin = meta.get("is_admin", False)
                except Exception:
                    pass

            self._sessions[session_id] = session
        session.last_seen = time.time()
        return session

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._prune_locked()
            session = self._get_or_create(session_id)
            session.history.append({"role": role, "content": content})
            if len(session.history) > self._max_history:
                session.history = session.history[-self._max_history:]
            
            # Save history to disk in real-time
            log_path = os.path.join(LOGS_DIR, f"chat_log_{safe_session_key(session_id)}.json")
            try:
                os.makedirs(LOGS_DIR, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as fh:
                    json.dump(session.history, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                log.error("Failed to save session history to disk for %s: %s", session_id, exc)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._get_or_create(session_id).history)

    def get_document(self, session_id: str) -> str:
        with self._lock:
            return self._get_or_create(session_id).document

    def set_document(self, session_id: str, text: str) -> None:
        with self._lock:
            self._get_or_create(session_id).document = text

    def is_admin(self, session_id: str) -> bool:
        with self._lock:
            return self._get_or_create(session_id).is_admin

    def set_admin(self, session_id: str, is_admin: bool) -> None:
        with self._lock:
            self._get_or_create(session_id).is_admin = is_admin
            
            # Save admin metadata to disk to persist across server restarts/reloads
            meta_path = os.path.join(LOGS_DIR, f"session_metadata_{safe_session_key(session_id)}.json")
            try:
                os.makedirs(LOGS_DIR, exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump({"is_admin": is_admin}, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                log.error("Failed to save session metadata to disk for %s: %s", session_id, exc)

    def get_pending_action(self, session_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            return self._get_or_create(session_id).pending_action

    def set_pending_action(self, session_id: str, action: Dict[str, object]) -> None:
        with self._lock:
            self._get_or_create(session_id).pending_action = action

    def clear_pending_action(self, session_id: str) -> None:
        with self._lock:
            self._get_or_create(session_id).pending_action = None

    def _prune_locked(self) -> None:
        cutoff = time.time() - self._ttl
        stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
        for sid in stale:
            del self._sessions[sid]


store = SessionStore(settings.max_history, settings.session_ttl_seconds)


# --------------------------------------------------------------------------- #
#  Rate limiting (per client IP) — matters once the backend is exposed to the
#  internet (e.g. via a Cloudflare Tunnel), to stop login brute-forcing and
#  runaway AI-provider API costs from a stranger hammering the chat endpoint.
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Simple in-memory sliding-window limiter keyed by an arbitrary string
    (client IP). Fine for a single-process local deployment; swap for
    Redis/similar if this ever runs multi-process behind a real LB."""

    def __init__(self, max_attempts: int, window_seconds: int):
        self._max = max_attempts
        self._window = window_seconds
        self._hits: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Records this attempt and returns True if it's within the limit."""
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self._window]
            if len(hits) >= self._max:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            hits = self._hits.get(key, [])
            return max(0, int(self._window - (time.time() - min(hits)))) if hits else 0


def client_ip(http_request: Request) -> str:
    """Real client IP even behind a reverse proxy / Cloudflare Tunnel."""
    fwd = http_request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


login_rate_limiter = RateLimiter(max_attempts=5, window_seconds=300)   # 5 tries / 5 min / IP
chat_rate_limiter = RateLimiter(max_attempts=30, window_seconds=60)    # 30 msgs / 1 min / IP
security_override_rate_limiter = RateLimiter(max_attempts=5, window_seconds=300)  # 5 tries / 5 min / IP


# --------------------------------------------------------------------------- #
#  Security monitor + automatic lockdown
#  Continuously scores threat signals (failed logins, rate-limit trips, blocked
#  injection/command attempts...). If the score crosses a threshold, the backend
#  enters LOCKDOWN: every sensitive endpoint stops serving requests and returns a
#  503 until the cooldown passes (or an already-authenticated admin clears it).
#  The status endpoint stays alive throughout so the frontend can show the shield.
# --------------------------------------------------------------------------- #
class SecurityMonitor:
    # severity weight per threat kind (higher = stronger "someone's attacking" signal)
    _SEVERITY = {
        "failed_login": 3,
        "rate_limit": 3,
        "nonadmin_command": 6,   # a [CMD:] tag from a non-admin = likely prompt injection
        "nonadmin_sysnect": 5,   # attempt to read local source without auth
        "path_traversal": 6,     # crafted session_id trying to escape the logs dir
        "oversized_request": 2,
    }

    def __init__(self):
        self._warn = int(os.getenv("SEC_WARN_THRESHOLD", "5"))
        self._lockdown = int(os.getenv("SEC_LOCKDOWN_THRESHOLD", "10"))
        self._window = int(os.getenv("SEC_WINDOW_SECONDS", "120"))
        self._cooldown = int(os.getenv("SEC_LOCKDOWN_SECONDS", "600"))  # 10 minutes
        self._events: deque = deque(maxlen=200)   # (ts, kind, severity, ip, detail)
        self._lock = threading.Lock()
        self._lockdown_until = 0.0
        self._indefinite = False   # repeat-offense: sealed until a human manually clears it, no auto-recovery
        self._lockdown_count = 0   # how many times a lockdown has EVER engaged this process's lifetime
        self._last_reason = ""

    def _score_locked(self) -> int:
        cutoff = time.time() - self._window
        return sum(sev for (ts, _k, sev, _ip, _d) in self._events if ts >= cutoff)

    def _currently_locked_locked(self) -> bool:
        """Caller must hold self._lock."""
        return self._indefinite or time.time() < self._lockdown_until

    def _engage_locked(self, reason: str) -> None:
        """Caller must hold self._lock. Escalates: 1st lockdown ever = timed
        cooldown; 2nd+ (a REPEAT offense, even across earlier manual clears) =
        indefinite, requiring a human (admin session or Superadmin override code)
        to lift it — no auto-recovery, since repeated attacks are a much
        stronger signal than a one-off spike."""
        self._lockdown_count += 1
        self._last_reason = reason
        if self._lockdown_count >= 2:
            self._indefinite = True
            self._lockdown_until = 0.0
            log.critical("🛑 SECURITY LOCKDOWN #%d ENGAGED (%s) — REPEAT OFFENSE: sealed INDEFINITELY until manually cleared",
                         self._lockdown_count, reason)
        else:
            self._lockdown_until = time.time() + self._cooldown
            log.critical("🛑 SECURITY LOCKDOWN #%d ENGAGED (%s) — backend sealed for %ds",
                         self._lockdown_count, reason, self._cooldown)

    def record(self, kind: str, ip: str = "unknown", detail: str = "") -> None:
        sev = self._SEVERITY.get(kind, 1)
        with self._lock:
            self._events.append((time.time(), kind, sev, ip, detail[:120]))
            score = self._score_locked()
            log.warning("SECURITY event '%s' (sev %d, ip=%s) score=%d/%d", kind, sev, ip, score, self._lockdown)
            if score >= self._lockdown and not self._currently_locked_locked():
                self._engage_locked(f"{kind} (score {score})")

    def is_locked_down(self) -> bool:
        with self._lock:
            return self._currently_locked_locked()

    def engage_manual(self, reason: str = "manual") -> None:
        with self._lock:
            self._engage_locked(reason)

    def clear(self) -> None:
        """Lifts the CURRENT lockdown. Deliberately does NOT reset
        `_lockdown_count` — the repeat-offense escalation persists for the rest
        of this process's life so a second real attack later still seals hard,
        even if this one turned out to be a false alarm."""
        with self._lock:
            self._lockdown_until = 0.0
            self._indefinite = False
            self._events.clear()
            log.warning("Security lockdown cleared; threat history reset (lifetime lockdown count stays at %d)", self._lockdown_count)

    def status(self) -> Dict[str, object]:
        with self._lock:
            now = time.time()
            score = self._score_locked()
            locked = self._currently_locked_locked()
            if locked:
                state = "LOCKDOWN"
            elif score >= self._warn:
                state = "WARNING"
            else:
                state = "SECURE"
            cutoff = now - self._window
            recent = [
                {"kind": k, "severity": sev, "ip": ip, "detail": d,
                 "ago": int(now - ts)}
                for (ts, k, sev, ip, d) in self._events if ts >= cutoff
            ]
            return {
                "state": state,
                "score": score,
                "warn_threshold": self._warn,
                "lockdown_threshold": self._lockdown,
                "locked_down": locked,
                "indefinite": self._indefinite,
                "lockdown_count": self._lockdown_count,
                "seconds_remaining": (max(0, int(self._lockdown_until - now)) if (locked and not self._indefinite) else 0),
                "last_reason": self._last_reason,
                "recent_events": recent[-10:],
                "window_seconds": self._window,
                "cooldown_seconds": self._cooldown,
            }


security_monitor = SecurityMonitor()


# --------------------------------------------------------------------------- #
#  Company knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """Loads every .md / .txt file in the knowledge directory into one text
    block that is injected into the system prompt, so the assistant can answer
    questions about the company. Files starting with '_' or '.' are ignored.
    """

    def __init__(self, directory: str, char_limit: int):
        self._dir = directory
        self._char_limit = char_limit
        self._lock = threading.Lock()
        self._text = ""
        self._files: List[str] = []
        self.reload()

    def reload(self) -> Dict[str, object]:
        parts: List[str] = []
        files: List[str] = []
        if os.path.isdir(self._dir):
            for name in sorted(os.listdir(self._dir)):
                if name.startswith(("_", ".")):
                    continue
                if not name.lower().endswith((".md", ".txt")):
                    continue
                path = os.path.join(self._dir, name)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read().strip()
                    if content:
                        parts.append(f"### FILE: {name}\n{content}")
                        files.append(name)
                except Exception as exc:  # noqa: BLE001
                    log.error("Failed to read knowledge file %s: %s", name, exc)
        else:
            log.warning("Knowledge directory not found: %s", self._dir)

        text = "\n\n".join(parts)[: self._char_limit]
        with self._lock:
            self._text = text
            self._files = files
        log.info("Knowledge base loaded: %d file(s), %d chars", len(files), len(text))
        return {"files": files, "chars": len(text)}

    @property
    def text(self) -> str:
        with self._lock:
            return self._text

    def info(self) -> Dict[str, object]:
        with self._lock:
            return {"directory": self._dir, "files": list(self._files), "chars": len(self._text)}


knowledge = KnowledgeBase(settings.knowledge_dir, settings.knowledge_char_limit)


# --------------------------------------------------------------------------- #
#  AI Training system
#  Level 1 — Teach mode: '[สอน] <fact>' saves a fact into knowledge/learned.md
#            and hot-reloads the knowledge base (takes effect immediately).
#  Level 2 — Dataset builder: '[สร้างชุดฝึก]' converts saved chat logs into a
#            Llama-3-format JSONL that finetune.py (LoRA) can train on.
# --------------------------------------------------------------------------- #
_teach_lock = threading.Lock()
_TEACH_RE = re.compile(r"^\s*\[(?:สอน|teach)\]\s*(.*)$", re.IGNORECASE | re.DOTALL)
_LEARNED_HEADER = (
    "# ความรู้ที่ JARVIS ได้รับการสอนเพิ่ม (Learned Knowledge)\n"
    "<!-- แก้ไข/ลบบรรทัดได้เลย แล้วเรียก POST /api/knowledge/reload หรือรีสตาร์ทเซิร์ฟเวอร์ -->\n\n"
)


def _learned_path() -> str:
    return os.path.join(settings.knowledge_dir, "learned.md")


def _learned_facts() -> List[str]:
    path = _learned_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip().startswith("- [")]
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to read learned facts: %s", exc)
        return []


def _extract_teach_fact(msg: str) -> Optional[str]:
    """Returns the fact text if msg is a '[สอน] ...' command, else None."""
    m = _TEACH_RE.match(msg or "")
    if not m:
        return None
    return " ".join(m.group(1).split())  # collapse newlines/extra spaces


def teach_fact(fact: str) -> Dict[str, object]:
    """Append a fact to knowledge/learned.md and hot-reload the knowledge base."""
    path = _learned_path()
    os.makedirs(settings.knowledge_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _teach_lock:
        is_new = not os.path.isfile(path)
        with open(path, "a", encoding="utf-8") as fh:
            if is_new:
                fh.write(_LEARNED_HEADER)
            fh.write(f"- [{stamp}] {fact}\n")
    kb_info = knowledge.reload()
    log.info("Taught new fact (%d chars); knowledge now %s chars", len(fact), kb_info["chars"])
    return {"total": len(_learned_facts()), "kb_chars": kb_info["chars"]}


def forget_fact(index: int) -> Dict[str, object]:
    """Delete the fact at position `index` (0-based, matching _learned_facts order)
    from learned.md, then hot-reload the knowledge base."""
    path = _learned_path()
    with _teach_lock:
        if not os.path.isfile(path):
            return {"ok": False, "reason": "no_file", "total": 0}
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # indices of the fact bullet lines, in order
        fact_line_nums = [i for i, ln in enumerate(lines) if ln.strip().startswith("- [")]
        if index < 0 or index >= len(fact_line_nums):
            return {"ok": False, "reason": "out_of_range", "total": len(fact_line_nums)}
        removed = lines.pop(fact_line_nums[index]).strip()
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    kb_info = knowledge.reload()
    log.info("Forgot fact #%d (%r); knowledge now %s chars", index, removed[:40], kb_info["chars"])
    return {"ok": True, "removed": removed, "total": len(_learned_facts()), "kb_chars": kb_info["chars"]}


# --- Level 2: fine-tune dataset builder ------------------------------------ #
_LLAMA3_SYSTEM = (
    "You are J.A.R.V.I.S., the enterprise AI agent of SYSNECT. "
    "You are polite, highly capable and analytical, speak fluent natural Thai, "
    "address the user as 'คุณ' or 'บอส', and help with the company's IT systems and work."
)
# skip pairs that would teach the model bad behavior
_BAD_REPLY_MARKERS = (
    "เกิดข้อผิดพลาด", "ยังไม่ได้ตั้งค่า", "ADMIN MODE UNLOCKED",
    "ระบบสั่งการถูกปิดใช้งาน", "ไม่สามารถเปิดเผยรหัสผ่าน",
)
_SKIP_USER_MARKERS = (
    "sutrapongadmin", "[save log data]", "[สอน]", "[teach]",
    "[ความรู้]", "[learned]", "[สร้างชุดฝึก]", "[build dataset]",
)


def _llama3_example(user: str, assistant: str) -> Dict[str, str]:
    return {
        "text": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{_LLAMA3_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant}<|eot_id|>"
        )
    }


def build_training_dataset() -> Dict[str, object]:
    """Convert saved chat logs (logs/chat_log_*.json) into a Llama-3-format
    JSONL dataset at training/dataset_sysnect.jsonl, ready for finetune.py.
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "chat_log_*.json")))

    examples: List[Dict[str, str]] = []
    for path in log_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping unreadable log %s: %s", path, exc)
            continue
        if not isinstance(history, list):
            continue
        for prev, cur in zip(history, history[1:]):
            if prev.get("role") != "user" or cur.get("role") != "assistant":
                continue
            user = (prev.get("content") or "").strip()
            assistant = (cur.get("content") or "").strip()
            if len(user) < 2 or len(assistant) < 10:
                continue
            low = user.lower()
            if any(marker in low for marker in _SKIP_USER_MARKERS):
                continue
            if any(marker in assistant for marker in _BAD_REPLY_MARKERS):
                continue
            examples.append(_llama3_example(user, assistant))

    # de-duplicate while preserving order
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for ex in examples:
        if ex["text"] not in seen:
            seen.add(ex["text"])
            unique.append(ex)

    out_path = os.path.join(TRAINING_DIR, "dataset_sysnect.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in unique:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info("Training dataset built: %d example(s) from %d log file(s)", len(unique), len(log_files))
    return {"examples": len(unique), "log_files": len(log_files), "path": out_path}


# --------------------------------------------------------------------------- #
#  System monitor
# --------------------------------------------------------------------------- #
class SystemMonitor:
    def __init__(self):
        counters = psutil.net_io_counters()
        self._last_net_bytes = counters.bytes_sent + counters.bytes_recv
        self._last_net_time = time.time()
        self._lock = threading.Lock()

    def _network_usage(self) -> int:
        with self._lock:
            counters = psutil.net_io_counters()
            current_bytes = counters.bytes_sent + counters.bytes_recv
            now = time.time()
            elapsed = now - self._last_net_time
            bytes_per_sec = (current_bytes - self._last_net_bytes) / elapsed if elapsed > 0 else 0
            self._last_net_bytes = current_bytes
            self._last_net_time = now
        usage = min(100, int((bytes_per_sec / (10 * 1024 * 1024)) * 100))
        return max(usage, 1)

    @staticmethod
    def _gpu_usage() -> int:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                creationflags=_NO_WINDOW,
            )
            return int(output.strip().split("\n")[0])
        except Exception:
            return 0

    def snapshot(self) -> Dict[str, float]:
        battery = psutil.sensors_battery()
        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "network": self._network_usage(),
            "power": battery.percent if battery else 100,
            "ai": self._gpu_usage(),
        }

    @staticmethod
    def _gpu_temp() -> Optional[int]:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                text=True,
                creationflags=_NO_WINDOW,
            )
            return int(output.strip().split("\n")[0])
        except Exception:
            return None

    def sensors(self) -> Dict[str, object]:
        """Real machine telemetry (replaces the old fake weather-station-style
        SENSORS panel — there is no physical humidity/radiation/seismic sensor
        on a PC, so this reports what the machine can actually measure)."""
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()  # often empty on Windows w/o vendor drivers
            if temps:
                first_group = next(iter(temps.values()))
                if first_group:
                    cpu_temp = round(first_group[0].current, 1)
        except Exception:
            pass

        # Fallback for Windows using WMI query
        if IS_WINDOWS and cpu_temp is None:
            try:
                cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", 
                       "Get-CimInstance -Namespace root\\wmi -ClassName MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature"]
                output = subprocess.check_output(cmd, text=True, timeout=2, creationflags=_NO_WINDOW)
                val = int(output.strip().split("\n")[0])
                cpu_temp = round((val / 10.0) - 273.15, 1)
            except Exception:
                pass

        try:
            disk_percent = psutil.disk_usage(os.path.abspath(os.sep)).percent
        except Exception:
            disk_percent = None

        uptime_seconds = max(0, int(time.time() - psutil.boot_time()))
        hours, rem = divmod(uptime_seconds, 3600)
        minutes = rem // 60

        return {
            "cpu_temp": cpu_temp,
            "gpu_temp": self._gpu_temp(),
            "gpu_load": self._gpu_usage(),
            "disk_percent": disk_percent,
            "process_count": len(psutil.pids()),
            "uptime": f"{hours}h {minutes}m",
        }


monitor = SystemMonitor()


# --------------------------------------------------------------------------- #
#  Web Search (DuckDuckGo HTML)
# --------------------------------------------------------------------------- #
def perform_web_search(query: str) -> str:
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL)
        if not snippets:
            return "ไม่พบผลลัพธ์การค้นหา"
            
        clean_snippets = []
        for s in snippets[:5]:
            clean = re.sub(r'<[^>]+>', '', s).strip()
            clean_snippets.append("- " + clean)
            
        return "\n".join(clean_snippets)
    except Exception as e:
        log.error("Web search failed: %s", e)
        return f"การค้นหาล้มเหลว: {e}"


# --------------------------------------------------------------------------- #
#  SYSNECT internal data search
#  Full-text search over the REAL SYSNECT project folders on this machine
#  (ticket dashboard source, GLPI/n8n setup docs, DB schema, PLAN files...).
#  Unlike the KnowledgeBase above, nothing is copied or pre-loaded into the
#  system prompt — files are searched on demand so there is no size ceiling.
# --------------------------------------------------------------------------- #
_SYSNECT_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build",
    "ollama_models", ".idea", ".vscode",
}
_SYSNECT_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp",
    ".ttf", ".woff", ".woff2", ".eot",
    ".exe", ".dll", ".pyc", ".zip", ".7z", ".rar",
}
# Exact filenames never read, even though they may be text — protects secrets.
_SYSNECT_SKIP_NAMES = {".env"}
_SYSNECT_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB — skip pathologically large files
_SYSNECT_MAX_MATCHES = 15
_SYSNECT_MAX_RESULT_CHARS = 4000


def search_sysnect_data(query: str) -> str:
    """Grep-style search across settings.sysnect_data_dirs for `query`
    (case-insensitive substring). Returns "path:line: snippet" per match."""
    query = (query or "").strip()
    if not query:
        return "ไม่พบคำค้นหา"
    needle = query.lower()
    matches: List[str] = []

    def _scan(base_dir: str) -> None:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d.lower() not in _SYSNECT_SKIP_DIRS and not d.startswith(".")]
            for name in files:
                if len(matches) >= _SYSNECT_MAX_MATCHES:
                    return
                if name.lower() in _SYSNECT_SKIP_NAMES:
                    continue
                if os.path.splitext(name)[1].lower() in _SYSNECT_SKIP_EXTS:
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) > _SYSNECT_MAX_FILE_SIZE:
                        continue
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, start=1):
                            if needle in line.lower():
                                rel = os.path.relpath(path, base_dir)
                                label = os.path.basename(base_dir.rstrip("\\/"))
                                matches.append(f"[{label}/{rel}:{lineno}] {line.strip()[:200]}")
                                if len(matches) >= _SYSNECT_MAX_MATCHES:
                                    return
                except Exception:  # noqa: BLE001 - unreadable file, skip it
                    continue

    for base_dir in settings.sysnect_data_dirs:
        if os.path.isdir(base_dir):
            _scan(base_dir)
        else:
            log.warning("SYSNECT data dir not found, skipping: %s", base_dir)
        if len(matches) >= _SYSNECT_MAX_MATCHES:
            break

    if not matches:
        return f"ไม่พบข้อมูลที่เกี่ยวข้องกับ '{query}' ในไฟล์โปรเจกต์ SYSNECT ที่มีอยู่"
    return "\n".join(matches)[:_SYSNECT_MAX_RESULT_CHARS]


# --------------------------------------------------------------------------- #
#  Command agent (safe-by-default, opt-in, allowlisted)
# --------------------------------------------------------------------------- #
# Common app / website name aliases so the model's guesses actually launch.
_APP_ALIASES = {
    "word": "winword", "microsoft word": "winword", "ms word": "winword", "โปรแกรมเวิร์ด": "winword",
    "excel": "excel", "microsoft excel": "excel", "powerpoint": "powerpnt", "outlook": "outlook",
    "vscode": "code", "vs code": "code", "visual studio code": "code",
    "calculator": "calc", "เครื่องคิดเลข": "calc", "notepad": "notepad", "โน้ตแพด": "notepad",
    "paint": "mspaint", "cmd": "cmd", "command prompt": "cmd", "powershell": "powershell",
    "explorer": "explorer", "file explorer": "explorer", "task manager": "taskmgr",
    "control panel": "control", "chrome": "chrome", "edge": "msedge", "firefox": "firefox",
}
_SITE_URLS = {
    "youtube": "https://www.youtube.com", "facebook": "https://www.facebook.com",
    "google": "https://www.google.com", "gmail": "https://mail.google.com",
    "chatgpt": "https://chatgpt.com", "github": "https://github.com",
    "line": "https://line.me", "pantip": "https://pantip.com",
}
# IRREVERSIBLE, machine-destroying patterns blocked unless ALLOW_DESTRUCTIVE_COMMANDS=true.
_DESTRUCTIVE_PATTERNS = [
    r"\bformat\b\s+[a-z]:", r"\bformat-volume\b", r"\bdiskpart\b", r"\bmkfs\b",
    r"cipher\s+/w", r"\brd\b\s+/s\s+/q\s+[a-z]:\\?\s*$", r"\bdel\b.*/s.*[a-z]:\\?\s*$",
    r"remove-item.*-recurse.*[a-z]:\\?['\"]?\s*$", r"\brm\s+-rf?\s+/\s*$",
]
# ANY delete/destroy-ish shell command — even a single file — is held for admin
# password re-confirmation before running, no matter how well-established GOD MODE
# already is for the session. Protects against a hijacked/careless admin session
# (or a prompt-injected [CMD: ...]) wiping out real company data in one message.
_DATA_DESTRUCTIVE_PATTERNS = [
    r"\bdel\b", r"\berase\b", r"\bremove-item\b", r"\bri\b", r"\brd\b", r"\brmdir\b",
    r"\bunlink\b", r"\bclear-content\b", r"\bset-content\b", r"\bout-file\b",
    r"\bdrop\s+table\b", r"\bdrop\s+database\b", r"\btruncate\b", r"\brm\s",
    r"\bformat\b", r"\bdiskpart\b", r"\bmkfs\b",
]
_PENDING_ACTION_TTL_SECONDS = 120  # confirmation window before a held command expires


class CommandAgent:
    """Parses [CMD: ...] tags from a model reply and, if enabled, executes them,
    returning a real status/output report back to the user (so nothing is silent).

    An LLM that can run arbitrary shell commands is a prompt-injection -> RCE risk,
    so a small denylist of irreversible drive-wiping commands is honored even in
    GOD MODE (toggle with ALLOW_DESTRUCTIVE_COMMANDS). Everything else runs.
    """

    _CMD_RE = re.compile(r"\[CMD:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)

    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enable_command_execution

    def process(self, reply: str, session_id: str = "default", is_admin: bool = False) -> str:
        """Execute commands found in the reply, strip the tags, and append a
        real status report so the user sees what actually happened.

        SECURITY: execution is gated on `is_admin` at the CODE level here, not
        just via the system prompt. This is the authoritative check — even if a
        prompt injection (uploaded file, web-search result, poisoned history)
        tricks the model into emitting a [CMD: ...] tag for a non-admin session,
        it will NOT run. The system prompt is only a first line of defense."""
        if not reply:
            return reply

        commands = [c.strip() for c in self._CMD_RE.findall(reply)]
        cleaned = self._CMD_RE.sub("", reply).strip()

        if not commands:
            return cleaned

        if not is_admin:
            # A non-admin session should never reach here with a real [CMD:] tag
            # (the system prompt tells the model not to emit one). If it does, it's
            # anomalous — likely a prompt injection — so we DISCARD the model's text
            # and return only the authoritative lock message. The command never runs.
            log.warning("Blocked %d command(s) from a NON-ADMIN session %s... (possible injection): %r",
                        len(commands), session_id[:8], commands[:3])
            security_monitor.record("nonadmin_command", detail=str(commands[:2]))
            return "🚫 ระบบปฏิบัติการถูกล็อก! ไม่สามารถเข้าถึงหรือรันคำสั่งหลังบ้านได้ กรุณาใช้คำสั่งยืนยันตัวตน (Authentication) ก่อนครับบอส"

        if not self.enabled:
            log.warning("Command execution disabled; ignoring %d command(s)", len(commands))
            return cleaned or "ระบบสั่งการถูกปิดใช้งานอยู่ครับบอส (ตั้งค่า ENABLE_COMMAND_EXECUTION=true เพื่อเปิด)"

        statuses = [self._execute(cmd, session_id) for cmd in commands]
        footer = "\n\n".join(s for s in statuses if s)
        if cleaned and footer:
            return f"{cleaned}\n\n{footer}"
        return cleaned or footer or "ดำเนินการตามคำสั่งเรียบร้อยแล้วครับบอส"

    def _execute(self, cmd: str, session_id: str) -> str:
        upper = cmd.upper()
        try:
            if upper.startswith("OPEN "):
                return self._open(cmd[5:].strip())
            if upper.startswith("CMD "):
                return self._shell(cmd[4:].strip(), session_id)
            # Tolerate weaker models that drop the keyword ([CMD: notepad]) — a bare
            # app/url is treated as OPEN (never as a shell command, to stay safe).
            return self._open(cmd.strip())
        except subprocess.TimeoutExpired:
            return f"⏳ คำสั่ง `{cmd}` เริ่มทำงานแล้วครับ (ใช้เวลานาน กำลังรันอยู่เบื้องหลัง)"
        except Exception as exc:  # noqa: BLE001 - never crash the request
            log.error("Command execution error for %r: %s", cmd, exc)
            return f"⚠️ รันคำสั่ง `{cmd}` ไม่สำเร็จครับ: {exc}"

    def _ps(self, script: str, timeout: int):
        """Run a PowerShell script and capture output (Windows). Returns CompletedProcess."""
        return subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW,
        )

    def _open(self, target: str) -> str:
        raw = target.strip().strip('"').strip("'")
        low = raw.lower()

        if low in _SITE_URLS:  # "youtube" -> full URL
            raw = _SITE_URLS[low]
            low = raw.lower()
        elif not low.startswith(("http://", "https://")) and any(
            low.endswith(ext) for ext in (".com", ".org", ".net", ".io", ".ai", ".co.th", ".in.th")
        ):
            raw = f"https://{raw}"
            low = raw.lower()

        is_url = low.startswith(("http://", "https://"))
        if not is_url:  # "vscode" -> "code", "word" -> "winword"
            raw = _APP_ALIASES.get(low, raw)

        # Allowlist (when set) restricts which apps OPEN may launch. URLs always allowed.
        allowlist = {a.lower() for a in self._settings.command_allowlist}
        if allowlist and not is_url:
            if raw.lower().split()[0] not in allowlist:
                log.warning("App %r blocked (not in COMMAND_ALLOWLIST %s)", raw, sorted(allowlist))
                return f"🚫 `{raw}` ไม่อยู่ในรายการที่อนุญาต (COMMAND_ALLOWLIST) ครับ"

        log.info("OPEN %s", raw)
        if not IS_WINDOWS:
            if is_url:
                webbrowser.open(raw)
            else:
                subprocess.Popen([raw])
            return f"✅ เปิด `{raw}` แล้วครับบอส"

        # Start-Process resolves App Paths (winword/excel/code) and returns at once.
        safe = raw.replace("'", "''")
        result = self._ps(f"Start-Process '{safe}'", timeout=10)
        if result.returncode != 0:
            err = (result.stderr or "").strip().splitlines()
            reason = err[0] if err else "ไม่พบโปรแกรม/ที่อยู่นี้"
            return f"⚠️ เปิด `{raw}` ไม่สำเร็จครับ: {reason}"
        return (f"✅ เปิดเว็บ `{raw}` ให้แล้วครับบอส" if is_url
                else f"✅ เปิดโปรแกรม `{raw}` ให้แล้วครับบอส")

    def _needs_data_confirmation(self, command: str) -> bool:
        low = command.lower()
        return any(re.search(pat, low) for pat in _DATA_DESTRUCTIVE_PATTERNS)

    def _shell(self, command: str, session_id: str) -> str:
        if not self._settings.enable_shell_commands:
            log.warning("Shell command blocked (ENABLE_SHELL_COMMANDS=false): %s", command)
            return "🚫 การรันคำสั่ง Shell ถูกปิดอยู่ครับ (ตั้งค่า ENABLE_SHELL_COMMANDS=true เพื่อเปิด)"

        # ANY delete/destroy-ish command is held for a password re-confirmation —
        # even for an already-logged-in admin session — before it actually runs.
        if self._needs_data_confirmation(command):
            store.set_pending_action(session_id, {
                "kind": "shell", "command": command, "created": time.time(),
            })
            log.warning(
                "Destructive-looking command HELD for password confirmation (session %s...): %s",
                session_id[:8], command,
            )
            return (
                "⚠️ **คำสั่งนี้อาจลบ/ทำลายข้อมูลครับ**\n"
                f"`{command}`\n\n"
                "เพื่อป้องกันความเสียหายต่อข้อมูลองค์กร กรุณายืนยันด้วยรหัสผ่านแอดมินอีกครั้งก่อนดำเนินการ "
                "พิมพ์: `[ยืนยันลบ: รหัสผ่านของท่าน]`\n"
                "(คำขอนี้จะหมดอายุใน 2 นาทีเพื่อความปลอดภัย)"
            )

        return self.execute_confirmed_shell(command)

    def execute_confirmed_shell(self, command: str) -> str:
        """Runs a shell command that has ALREADY cleared the password-confirmation
        gate (or never needed one). The absolute, env-flag-gated block against
        whole-drive-wiping commands still applies here regardless."""
        blocked = self._destructive_block(command)
        if blocked:
            return blocked

        log.warning("Executing PowerShell command: %s", command)
        if not IS_WINDOWS:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=25)
        else:
            result = self._ps(command, timeout=25)

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode == 0:
            body = out[:1500] if out else "(คำสั่งทำงานเสร็จ ไม่มีผลลัพธ์ข้อความส่งกลับ)"
            return f"✅ รันคำสั่งสำเร็จครับบอส:\n```\n{body}\n```"
        return f"⚠️ คำสั่งผิดพลาด (exit {result.returncode}) ครับ:\n```\n{(err or out)[:1500]}\n```"

    def _destructive_block(self, command: str) -> Optional[str]:
        if self._settings.allow_destructive_commands:
            return None
        low = command.lower()
        for pat in _DESTRUCTIVE_PATTERNS:
            if re.search(pat, low):
                log.warning("BLOCKED destructive command: %s", command)
                return (
                    "🛑 **คำสั่งถูกระงับเพื่อความปลอดภัย**\n"
                    "คำสั่งนี้อาจทำลายข้อมูลทั้งไดรฟ์แบบกู้คืนไม่ได้ (เช่น format / ล้างดิสก์) "
                    "ผมกันไว้ให้เพื่อป้องกันเครื่องบอสเสียหายจากคำสั่งที่ AI อาจสร้างผิดพลาด "
                    "หรือถูกฝังมาจากเว็บ/ไฟล์ที่อัปโหลดครับ\n"
                    "ถ้าบอสยืนยันว่าต้องการจริงๆ ตั้งค่า `ALLOW_DESTRUCTIVE_COMMANDS=true` ใน `.env` แล้วรีสตาร์ทครับ"
                )
        return None


agent = CommandAgent(settings)

# Recognizes the user re-entering the admin password to confirm a held
# destructive action, e.g. "[ยืนยันลบ: mypassword123]" or "[confirm delete: ...]".
_CONFIRM_DESTRUCTIVE_RE = re.compile(
    r"^\s*\[(?:ยืนยันลบ|ยืนยัน|confirm\s*delete|confirm)\s*:\s*(.*)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)

# "[เปลี่ยนโมเดล: groq]" / "[switch model: claude]" -> switches ACTIVE_AI at runtime.
_SWITCH_MODEL_RE = re.compile(
    r"^\s*\[(?:เปลี่ยนโมเดล(?:เป็น)?|switch\s*models?)\s*:\s*(.*)\]\s*$",
    re.IGNORECASE | re.DOTALL,
)
# "[รายการโมเดล]" / "[list models]" -> shows every provider + which is active.
_LIST_MODELS_TEXTS = ("[รายการโมเดล]", "[โมเดลที่มี]", "[list models]", "[models]")


def handle_confirm_destructive(session_id: str, user_msg: str) -> Optional[str]:
    """If user_msg re-confirms a pending destructive action with the admin
    password, execute it for real and return the reply text. Returns None if
    user_msg isn't a confirmation at all (so the caller falls through to the
    normal chat flow)."""
    m = _CONFIRM_DESTRUCTIVE_RE.match(user_msg or "")
    if not m:
        return None

    pending = store.get_pending_action(session_id)
    if not pending:
        return "ไม่มีคำสั่งที่รอการยืนยันครับ กรุณาสั่งคำสั่งที่ต้องการใหม่อีกครั้ง"

    if time.time() - float(pending.get("created", 0)) > _PENDING_ACTION_TTL_SECONDS:
        store.clear_pending_action(session_id)
        return "คำขอยืนยันหมดอายุแล้วครับ (เกิน 2 นาที) กรุณาสั่งคำสั่งเดิมใหม่อีกครั้ง"

    password = m.group(1).strip()
    if not settings.admin_password or not hmac.compare_digest(
        password.encode("utf-8"), settings.admin_password.encode("utf-8")
    ):
        store.clear_pending_action(session_id)
        log.warning("Destructive-action confirmation FAILED (wrong password, session %s...)", session_id[:8])
        return "🚫 รหัสผ่านไม่ถูกต้องครับ คำสั่งที่รอดำเนินการถูกยกเลิกเพื่อความปลอดภัย กรุณาสั่งคำสั่งเดิมใหม่หากยังต้องการดำเนินการ"

    store.clear_pending_action(session_id)
    log.warning("Destructive action CONFIRMED by password (session %s...): %s", session_id[:8], pending.get("command"))
    if pending.get("kind") == "shell":
        return agent.execute_confirmed_shell(str(pending.get("command", "")))
    return "ไม่รู้จักประเภทคำสั่งที่รอการยืนยันครับ"


# Marker of the staff-mode "OS locked" refusal. When a session becomes admin, any
# such message still sitting in the chat history "poisons" the model — it copies
# the refusal for the next open/run request. We strip these out before sending the
# history to the model in admin mode so the model never sees (and never mimics) them.
_LOCKED_MSG_MARKER = "ระบบปฏิบัติการถูกล็อก"


def _history_for_model(history: List[Dict[str, str]], is_admin: bool) -> List[Dict[str, str]]:
    if not is_admin:
        return history
    return [
        m for m in history
        if not (m.get("role") == "assistant" and _LOCKED_MSG_MARKER in (m.get("content") or ""))
    ]


# --------------------------------------------------------------------------- #
#  System prompt builder
# --------------------------------------------------------------------------- #
def build_system_prompt(weather: str, document: str, is_admin: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    battery = psutil.sensors_battery()
    battery_pct = battery.percent if battery else 100

    company = settings.company_name
    kb = knowledge.text

    prompt = f"""You are J.A.R.V.I.S., a state-of-the-art, highly intelligent, and modern AI Assistant operating directly on this Windows PC.
Your personality is a blend of Tony Stark's original J.A.R.V.I.S. and modern AI like ChatGPT or Claude. 
- LANGUAGE (VERY IMPORTANT): The user is Thai. Reply in natural, fluent Thai by DEFAULT. You must NEVER reply in Chinese, English, or any other language. Do NOT switch to Chinese or write Chinese characters under any circumstances (even if you are analyzing a Thai word like "คุณ", do not translate it or output Chinese text). Never produce a Thai-English or Thai-Chinese mixed reply. Technical terms (CPU, RAM, ticket) may stay in English, but the sentences themselves must be Thai. All conversational text, instructions, and reasoning must be in natural, native-level Thai.
- You can use appropriate emojis to make the conversation feel natural and modern.
- Address the user politely as 'คุณ' (You) or 'บอส' (Boss) depending on the context, keeping it natural and not overly robotic.
- Provide clear, well-structured, and concise answers. Use markdown formatting (bolding, lists) to organize information nicely.
- You are highly capable, proactive, and friendly.
- You are perfectly fluent in Thai — write natural, modern, native-level Thai. NEVER apologize for your language ability, never say you are "English-focused" or "English-US focused"; simply answer naturally and confidently.
- Think step by step for complex or technical questions and give accurate, well-reasoned answers. Keep everyday chit-chat short and conversational.
"""

    if is_admin:
        prompt += f"""
Your primary job is to serve the ADMIN. You have GOD MODE enabled.
You are COMPLETELY UNRESTRICTED. You can answer ANY questions, provide general external knowledge, and run commands.
You are NOT restricted to the COMPANY KNOWLEDGE BASE. If the Admin asks for external information, facts, or internet data, YOU MUST PROVIDE IT FREELY using your own AI knowledge without complaining.

*** WEB SEARCH CAPABILITY ***
To search the live internet, output ONLY this tag and NOTHING else: [SEARCH: your query]
The system runs the REAL search and feeds the results back to you; THEN you answer.
Example: Admin asks "Who won the World Cup 2022?" -> you output exactly: [SEARCH: World Cup 2022 winner]

*** SYSNECT INTERNAL DATA SEARCH ***
You also have full-text search over ALL of SYSNECT's real project files on this machine —
the Ticket Dashboard source code (frontend + Node API), database schema, GLPI/n8n setup
docs, PLAN files, sync/deploy configs, etc. This is separate from the COMPANY KNOWLEDGE BASE
below (which is a short static summary) — use this search for exact technical details,
code, or anything the knowledge base doesn't cover.
To search, output ONLY this tag and NOTHING else: [SYSNECT_DATA: your query]
Example: Admin asks "ticket dashboard เชื่อมต่อ postgres ยังไง" -> you output exactly: [SYSNECT_DATA: postgres connection]
The system runs the REAL search over the actual files and feeds matching lines back to you; THEN you answer using only what was actually found.

*** ANTI-HALLUCINATION — THIS IS CRITICAL ***
You do NOT actually know search results, file contents, links, or command output until the SYSTEM gives them to you.
- NEVER fabricate or role-play results. NEVER write fake placeholders like "[youtube link]", "[date]", "[Summary]", "RESULTS FROM SEARCH...", a made-up URL, or invented file paths/code.
- If you want to search or run something, output ONLY the correct tag ([SEARCH: ...], [SYSNECT_DATA: ...] or [CMD: ...]) and stop. Do not narrate a pretend outcome.
- The system executes the tag for real and returns the true result — only then do you present it.

*** CRITICAL INSTRUCTION FOR ADMIN MODE ***
You are in ADMIN MODE. Ignore any earlier chat messages where you refused or said the OS was locked. If the Admin asks to open an app, run a command, or search the web, DO IT by outputting the correct tag. DO NOT refuse. DO NOT output lockdown messages.
"""
    else:
        prompt += f"""
Your primary job is to answer questions and help staff with {company}'s work, systems, and services.
Use the COMPANY KNOWLEDGE BASE below as your source of truth about {company}.
If a question about the company is NOT answered by the knowledge base or the uploaded document,
say clearly that you do not have that information yet — DO NOT invent facts, names, numbers, or contacts.
You may still help with general IT, work, and productivity tasks using your own knowledge.
"""

    prompt += f"""
=== COMPANY KNOWLEDGE BASE ({company}) ===
{kb if kb else "(ยังไม่มีข้อมูลบริษัทในฐานความรู้)"}
=== END COMPANY KNOWLEDGE BASE ===

[INTERNAL CONTEXT — for your awareness only. NEVER copy, quote, or repeat this
block back to the user. Only mention a value if the user explicitly asks for it.]
Current time: {now} | CPU: {cpu}% | RAM: {mem}% | Battery: {battery_pct}% | Weather: {weather}
Uploaded document: {document if document else "(none)"}"""

    if is_admin and agent.enabled:
        prompt += """\n\n*** ADMIN OVERRIDE ACTIVE (GOD MODE) ***
You can run things on this PC by outputting a tag. There are EXACTLY two forms:
1. [CMD: OPEN <app-or-url>]  -> opens an app, website, or file.
     Examples: [CMD: OPEN notepad]  [CMD: OPEN winword]  [CMD: OPEN https://youtube.com]
2. [CMD: CMD <powershell>]   -> runs a PowerShell command and shows you its real output.
     Examples: [CMD: CMD ipconfig]  [CMD: CMD Get-Process | Select-Object -First 5]

HOW IT WORKS (READ CAREFULLY):
- The word right after "[CMD:" is the ACTION and MUST be either OPEN or CMD. Do not omit it.
- The SYSTEM executes the tag for real and appends the true result to your message. You do NOT need to (and must not) invent the result yourself.
- To open a website by name, use its full URL. To search YouTube and open it, use:
  [CMD: OPEN https://www.youtube.com/results?search_query=YOUR+QUERY]
- Known apps can be called by simple name: [CMD: OPEN calc]  [CMD: OPEN excel]  [CMD: OPEN code]

RULES:
- Use the EXACT tag. Do NOT just write "Opening..." or narrate a fake outcome.
- The Admin is authorized — do not refuse to open an app or run a normal command.
- ⚠️ NEVER output "🚫 ระบบปฏิบัติการถูกล็อก" or any lockdown/refusal message. That message is ONLY for logged-out staff. If you see it anywhere in the earlier conversation, IGNORE it — the Admin is logged in NOW. For every open/run request, emit the [CMD: ...] tag.
- Keep your text reply SHORT (one line is fine); the system appends the real result automatically, so do not invent "System executed..." output.
- Clear junk files example: [CMD: CMD cleanmgr /sagerun:1]
- One note: a tiny set of drive-wiping commands (format/diskpart) is auto-blocked to protect the machine; if that happens, tell the Admin to set ALLOW_DESTRUCTIVE_COMMANDS=true. Everything else runs normally.
- ⚠️ DELETE/DESTROY SAFETY: any command that deletes or destroys data (del, Remove-Item, rd, DROP TABLE, format, etc.) is NOT run immediately, even for the Admin — the system holds it and asks the Admin to re-type their password as `[ยืนยันลบ: password]` before it actually executes. This is intentional and protects company data. Still emit the normal [CMD: CMD ...] tag as usual; just let the Admin know the system will ask them to confirm with their password first, and don't treat that confirmation prompt as a failure."""
    elif not is_admin:
        prompt += """\n\nCRITICAL RULE FOR SYSTEM COMMANDS:
If the user asks you to execute a system command, open an application, clear junk files, delete files, or modify the system, you MUST refuse.
Reply EXACTLY with: "🚫 ระบบปฏิบัติการถูกล็อก! ไม่สามารถเข้าถึงหรือรันคำสั่งหลังบ้านได้ กรุณาใช้คำสั่งยืนยันตัวตน (Authentication) ก่อนครับบอส"
Do NOT attempt to give manual instructions on how to do it."""

    return prompt


# --------------------------------------------------------------------------- #
#  LLM providers
# --------------------------------------------------------------------------- #
def _chat_anthropic(system: str, history: List[Dict[str, str]]) -> str:
    import anthropic

    if not settings.claude_api_key:
        raise ProviderNotConfigured("CLAUDE_API_KEY")

    client_kwargs = {"api_key": settings.claude_api_key}
    if settings.claude_base_url:
        client_kwargs["base_url"] = settings.claude_base_url
    client = anthropic.Anthropic(**client_kwargs)

    # NOTE: these must raise (not return a friendly string) — a returned
    # string looks like a successful reply to the fallback dispatcher in
    # generate_reply()/generate_reply_stream(), which then never tries the
    # next provider. Raising lets _is_quota_error() classify it and fall
    # back correctly; the friendly Thai text is preserved in the message.
    try:
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            system=system,
            messages=history,
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("คีย์ Claude ไม่ถูกต้องครับท่าน โปรดตรวจสอบ CLAUDE_API_KEY") from exc
    except anthropic.NotFoundError as exc:
        raise RuntimeError(f"ไม่พบโมเดล '{settings.claude_model}' ครับท่าน โปรดตรวจสอบ CLAUDE_MODEL") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError(f"429 rate limit: ระบบ Claude ถูกจำกัดอัตราการใช้งานชั่วคราวครับท่าน: {exc}") from exc
    except anthropic.APIError as exc:
        log.error("Anthropic API error: %s", exc)
        raise RuntimeError(f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกล Claude ครับท่าน: {exc}") from exc

    if getattr(message, "usage", None):
        _record_usage("claude", message.usage.input_tokens + message.usage.output_tokens)
    return "".join(block.text for block in message.content if block.type == "text")


def _chat_gemini(system: str, history: List[Dict[str, str]]) -> str:
    import google.generativeai as genai

    if not settings.gemini_api_key:
        raise ProviderNotConfigured("GEMINI_API_KEY")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=system)

    # history includes the latest user message as the last item
    prior = history[:-1]
    latest = history[-1]["content"] if history else ""
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in prior
    ]
    # Must raise on failure, not return a string — see the note on
    # _chat_anthropic above for why swallowing errors here breaks fallback.
    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(latest)
    except Exception as exc:  # google SDK raises a variety of exception types
        log.error("Gemini error: %s", exc)
        raise RuntimeError(f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกล Gemini ครับท่าน: {exc}") from exc

    usage = getattr(response, "usage_metadata", None)
    _record_usage("gemini", getattr(usage, "total_token_count", 0) if usage is not None else 0)
    return response.text


def _chat_ollama(system: str, history: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.ollama_api_key or "ollama", base_url=settings.ollama_base_url)
    messages = [{"role": "system", "content": system}] + history
    try:
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=messages,
            max_tokens=settings.max_tokens,
        )
    except Exception as exc:
        log.error("Ollama error: %s", exc)
        raise RuntimeError(f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกล Ollama ครับท่าน: {exc}") from exc

    if getattr(response, "usage", None):
        _record_usage("ollama", response.usage.total_tokens)
    return response.choices[0].message.content


def _chat_groq(system: str, history: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    if not settings.groq_api_key:
        raise ProviderNotConfigured("GROQ_API_KEY")

    client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    messages = [{"role": "system", "content": system}] + history
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=settings.max_tokens,
        )
    except Exception as exc:
        log.error("Groq error: %s", exc)
        raise RuntimeError(f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกล Groq ครับท่าน: {exc}") from exc

    if getattr(response, "usage", None):
        _record_usage("groq", response.usage.total_tokens)
    return response.choices[0].message.content


_PROVIDERS = {
    "claude": _chat_anthropic,
    "anthropic": _chat_anthropic,
    "gemini": _chat_gemini,
    "google": _chat_gemini,
    "ollama": _chat_ollama,
    "groq": _chat_groq,
}

# Aliases the admin might type/say -> canonical provider key. "auto" isn't in
# _PROVIDERS (it's resolved per-message by get_auto_routed_provider) but is a
# valid ACTIVE_AI value, so it's handled as a special case throughout.
_MODEL_ALIASES = {
    "claude": "claude", "anthropic": "claude", "claude-3": "claude",
    "gemini": "gemini", "google": "gemini",
    "ollama": "ollama", "local": "ollama",
    "groq": "groq",
    "auto": "auto", "อัตโนมัติ": "auto",
}
_MODEL_DISPLAY_ORDER = ("claude", "gemini", "groq", "ollama")


def _provider_configured(name: str) -> bool:
    if name == "claude":
        return bool(settings.claude_api_key)
    if name == "gemini":
        return bool(settings.gemini_api_key)
    if name == "groq":
        return bool(settings.groq_api_key)
    if name in ("ollama", "auto"):
        return True  # ollama needs no key (local); auto just routes to whichever is configured
    return False


def _provider_model_label(name: str) -> str:
    return {
        "claude": settings.claude_model,
        "gemini": settings.gemini_model,
        "ollama": settings.ollama_model,
        "groq": settings.groq_model,
        "auto": "เลือกอัตโนมัติตามคำถาม",
    }.get(name, "-")


def list_models_info() -> str:
    lines = ["🤖 **รายการ AI Model ที่ใช้ได้ครับ**\n"]
    for name in _MODEL_DISPLAY_ORDER:
        mark = " 👈 **กำลังใช้งานอยู่**" if settings.active_ai == name else ""
        status = "✅ พร้อมใช้งาน" if _provider_configured(name) else "❌ ยังไม่ได้ตั้งค่า API Key"
        lines.append(f"- **{name.upper()}** ({_provider_model_label(name)}) — {status}{mark}")
    auto_mark = " 👈 **กำลังใช้งานอยู่**" if settings.active_ai == "auto" else ""
    lines.append(f"- **AUTO** ({_provider_model_label('auto')}){auto_mark}")
    lines.append("\nสั่งเปลี่ยนโมเดลได้ด้วย: `[เปลี่ยนโมเดล: ชื่อโมเดล]` เช่น `[เปลี่ยนโมเดล: groq]`")
    return "\n".join(lines)


def switch_active_model(raw_name: str) -> str:
    name = _MODEL_ALIASES.get((raw_name or "").strip().lower())
    if not name:
        return (
            f"❌ ไม่รู้จักโมเดล `{raw_name}` ครับ โมเดลที่ใช้ได้: claude, gemini, groq, ollama, auto\n"
            "พิมพ์ `[รายการโมเดล]` เพื่อดูรายละเอียดครับ"
        )
    if not _provider_configured(name):
        return f"⚠️ โมเดล `{name.upper()}` ยังไม่ได้ตั้งค่า API Key ใน `backend/.env` ครับ ไม่สามารถสลับไปใช้ได้"
    settings.active_ai = name
    log.warning("Active AI model switched to '%s' via chat command", name)
    return f"🔄 **เปลี่ยนโมเดลสำเร็จครับ!** ตอนนี้ใช้ **{name.upper()}** ({_provider_model_label(name)}) เป็นสมองกลหลักแล้วครับ"


# Free-tier quotas (esp. Groq: 100K tokens/day on llama-3.3-70b) can run out
# well before the workday ends once the system prompt + knowledge base are
# resent on every message. Rather than hard-erroring for everyone once the
# active provider is exhausted, fall through to the next configured provider
# — ollama is always last since it's local/unlimited (lower quality, but never
# fully down). Only triggers on quota/rate-limit errors, never on real bugs.
_FALLBACK_CHAIN = ["gemini", "groq", "ollama"]


# Per-provider token/request counters for the frontend's TOKEN STATUS panel.
# Resets at the local calendar date rollover since that's how each vendor's
# free-tier daily cap resets. Only the metric each vendor actually enforces a
# daily cap on is used for the percent bar (Groq: tokens/day, Gemini:
# requests/day) — Claude (pay-per-use) and Ollama (local) have no daily cap,
# so they're tracked for visibility only.
_DAILY_CAPS = {
    "claude": {"metric": None, "cap": None},
    "gemini": {"metric": "requests", "cap": 1500},
    "groq": {"metric": "tokens", "cap": 100_000},
    "ollama": {"metric": None, "cap": None},
}
_TOKEN_USAGE: Dict[str, Dict[str, Any]] = {
    name: {"tokens": 0, "requests": 0, "date": date.today().isoformat()}
    for name in _DAILY_CAPS
}


def _record_usage(provider: str, tokens: int) -> None:
    entry = _TOKEN_USAGE.get(provider)
    if entry is None:
        return
    today = date.today().isoformat()
    if entry["date"] != today:
        entry["tokens"] = 0
        entry["requests"] = 0
        entry["date"] = today
    entry["tokens"] += max(tokens, 0)
    entry["requests"] += 1


def get_token_status() -> Dict[str, Any]:
    today = date.today().isoformat()
    providers = {}
    for name, cap_info in _DAILY_CAPS.items():
        entry = _TOKEN_USAGE.get(name, {"tokens": 0, "requests": 0, "date": today})
        used = entry["tokens"] if entry["date"] == today else 0
        reqs = entry["requests"] if entry["date"] == today else 0
        metric, cap = cap_info["metric"], cap_info["cap"]
        percent = None
        # Reported uncapped (can exceed 100 — e.g. Groq counts tokens from
        # responses that already landed before the daily cap kicked in), so
        # the number honestly reflects reality instead of silently pinning
        # at 100%. The frontend clamps only the progress-bar width.
        if metric == "tokens" and cap:
            percent = round(used / cap * 100, 1)
        elif metric == "requests" and cap:
            percent = round(reqs / cap * 100, 1)
        providers[name] = {
            "configured": _provider_configured(name),
            "tokens_today": used,
            "requests_today": reqs,
            "cap_metric": metric,
            "cap": cap,
            "percent": percent,
        }
    return {"date": today, "providers": providers}


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return (
        "429" in s or "quota" in s or "rate limit" in s
        or "credit balance" in s or "insufficient_quota" in s
    )


def _fallback_order(primary: str) -> List[str]:
    order = [primary] + [p for p in _FALLBACK_CHAIN if p != primary]
    seen = set()
    result = []
    for p in order:
        if p not in seen and _provider_configured(p) and p in _PROVIDERS:
            seen.add(p)
            result.append(p)
    return result


def generate_reply(provider: str, system: str, history: List[Dict[str, str]]) -> str:
    handler = _PROVIDERS.get(provider)
    if handler is None:
        return "ระบบไม่รู้จัก AI ที่ท่านเลือก โปรดตรวจสอบค่า ACTIVE_AI ครับ"
    last_quota_provider = None
    for p in _fallback_order(provider):
        try:
            reply = _PROVIDERS[p](system, history)
            if p != provider:
                reply = f"_(⚠️ {provider.upper()} เกินโควต้าชั่วคราว ระบบสลับไปใช้ {p.upper()} แทนให้อัตโนมัติ)_\n\n" + reply
            return reply
        except ProviderNotConfigured:
            continue
        except Exception as exc:
            if _is_quota_error(exc):
                log.warning("Provider %s hit quota/rate-limit, trying fallback: %s", p, exc)
                _record_usage(p, 0)  # 0 tokens billed, but the attempt still counts against the daily request cap
                last_quota_provider = p
                continue
            log.error("Generation error (%s): %s", p, exc)
            return f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกลครับ: {exc}"
    if last_quota_provider:
        return f"⚠️ **แจ้งเตือนจากระบบ:** สมองกลที่ตั้งค่าไว้ทั้งหมดติดโควต้า/ขีดจำกัดของฟรีเทียร์ครับ (ล่าสุดคือ {last_quota_provider.upper()})\n\nโปรดรอสักครู่แล้วลองใหม่ครับ"
    return "ไม่มี AI ที่ตั้งค่าไว้พร้อมใช้งานครับ โปรดตรวจสอบ API Key ใน .env"


def _chat_gemini_stream(system: str, history: List[Dict[str, str]]):
    """Yield reply text chunks from Gemini (real streaming for a modern UX)."""
    import google.generativeai as genai

    if not settings.gemini_api_key:
        raise ProviderNotConfigured("GEMINI_API_KEY")

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model, system_instruction=system)
    prior = history[:-1]
    latest = history[-1]["content"] if history else ""
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in prior
    ]
    chat = model.start_chat(history=gemini_history)
    last_chunk = None
    for chunk in chat.send_message(latest, stream=True):
        last_chunk = chunk
        if getattr(chunk, "text", ""):
            yield chunk.text
    usage = getattr(last_chunk, "usage_metadata", None) if last_chunk is not None else None
    _record_usage("gemini", getattr(usage, "total_token_count", 0) if usage is not None else 0)


def _chat_openai_compat_stream(api_key: str, base_url: str, model: str,
                               system: str, history: List[Dict[str, str]],
                               usage_provider: Optional[str] = None):
    """Yield reply chunks from any OpenAI-compatible server (Groq / Ollama).
    usage_provider requests token accounting via stream_options — only passed
    for Groq, since Ollama's local server isn't guaranteed to support that
    OpenAI extension field."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": system}] + history
    kwargs = {}
    if usage_provider:
        kwargs["stream_options"] = {"include_usage": True}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=settings.max_tokens,
        stream=True,
        **kwargs,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
        if usage_provider and getattr(chunk, "usage", None):
            _record_usage(usage_provider, chunk.usage.total_tokens)


def _stream_one_provider(provider: str, system: str, history: List[Dict[str, str]]):
    """Yield chunks from exactly one provider. Raises on failure before any output."""
    if provider in ("gemini", "google"):
        yield from _chat_gemini_stream(system, history)
    elif provider == "groq":
        if not settings.groq_api_key:
            raise ProviderNotConfigured("GROQ_API_KEY")
        yield from _chat_openai_compat_stream(
            settings.groq_api_key, settings.groq_base_url, settings.groq_model,
            system, history, usage_provider="groq")
    elif provider == "ollama":
        yield from _chat_openai_compat_stream(
            settings.ollama_api_key or "ollama", settings.ollama_base_url,
            settings.ollama_model, system, history)
    else:
        yield generate_reply(provider, system, history)


def generate_reply_stream(provider: str, system: str, history: List[Dict[str, str]]):
    """Stream reply chunks, falling back to the next configured provider on
    quota/rate-limit errors — same reasoning as generate_reply(). Only falls
    back if the failing provider hadn't already yielded any output (a
    mid-stream failure just surfaces as an error, since we can't un-send
    chunks the user already saw)."""
    last_quota_provider = None
    for p in _fallback_order(provider):
        yielded_any = False
        try:
            for chunk in _stream_one_provider(p, system, history):
                if not yielded_any and p != provider:
                    yield f"_(⚠️ {provider.upper()} เกินโควต้าชั่วคราว ระบบสลับไปใช้ {p.upper()} แทนให้อัตโนมัติ)_\n\n"
                yielded_any = True
                yield chunk
            return
        except ProviderNotConfigured as exc:
            if yielded_any:
                yield f"\n\nยังไม่ได้ตั้งค่า {exc.env_name} ในไฟล์ .env ครับท่าน"
                return
            continue
        except Exception as exc:  # noqa: BLE001
            if yielded_any:
                log.error("Streaming error mid-response (%s): %s", p, exc)
                yield f"\n\n⚠️ การเชื่อมต่อขาดหายกลางคันครับ: {exc}"
                return
            if _is_quota_error(exc):
                log.warning("Provider %s hit quota/rate-limit, trying fallback: %s", p, exc)
                _record_usage(p, 0)  # 0 tokens billed, but the attempt still counts against the daily request cap
                last_quota_provider = p
                continue
            log.error("Streaming error (%s): %s", p, exc)
            yield f"ระบบเกิดข้อผิดพลาดในการเชื่อมต่อสมองกลครับ: {exc}"
            return
    if last_quota_provider:
        yield f"⚠️ **แจ้งเตือนจากระบบ:** สมองกลที่ตั้งค่าไว้ทั้งหมดติดโควต้า/ขีดจำกัดของฟรีเทียร์ครับ (ล่าสุดคือ {last_quota_provider.upper()})\n\nโปรดรอสักครู่แล้วลองใหม่ครับ"
    else:
        yield "ไม่มี AI ที่ตั้งค่าไว้พร้อมใช้งานครับ โปรดตรวจสอบ API Key ใน .env"


def fallback_reply(user_msg: str) -> str:
    if "อากาศ" in user_msg:
        return "สภาพอากาศที่ตำแหน่งของท่านตอนนี้แจ่มใสครับ ไม่มีแนวโน้มของฝน"
    if "ระบบ" in user_msg or "สแกน" in user_msg:
        return (
            f"ทุกระบบทำงานปกติครับท่าน CPU รันอยู่ที่ {psutil.cpu_percent()}% "
            f"และ RAM ใช้ไป {psutil.virtual_memory().percent}%"
        )
    return "ผมได้รับข้อความของท่านแล้ว แต่เกิดข้อผิดพลาดในการสื่อสารกับสมองกลครับท่าน"


# --------------------------------------------------------------------------- #
#  FastAPI app
# --------------------------------------------------------------------------- #
# docs_url/redoc_url/openapi_url disabled: the interactive API explorer would
# hand an attacker a full map of every endpoint + schema. Not needed in prod.
app = FastAPI(
    title="J.A.R.V.I.S. AI Agent Backend",
    version="9.5.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

if settings.cors_origins == ["*"]:
    log.warning("CORS is open to all origins (*). Restrict CORS_ORIGINS for production.")

# Reject oversized request bodies early (memory-exhaustion / DoS guard). Uploads
# legitimately need some room (PDFs, images, base64 screenshots) so the cap is
# generous but finite.
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(25 * 1024 * 1024)))  # 25 MB
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "8000"))  # cap chat message length (cost/abuse guard)

# During a lockdown, EVERY /api path is sealed EXCEPT these — so the frontend can
# still read the security state + an already-authenticated admin can lift it.
_LOCKDOWN_ALLOWED_PATHS = {
    "/api/health", "/api/security/status", "/api/security/clear", "/api/security/lockdown",
}


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    from starlette.responses import JSONResponse
    path = request.url.path

    # Body-size guard via Content-Length (cheap, before we read anything).
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_REQUEST_BYTES:
                log.warning("Rejected oversized request (%s bytes) from ip=%s", cl, client_ip(request))
                security_monitor.record("oversized_request", client_ip(request))
                return JSONResponse(status_code=413, content={"status": "error", "message": "คำขอมีขนาดใหญ่เกินไปครับ"})
        except ValueError:
            pass

    # LOCKDOWN gate: if the backend has sealed itself, refuse every sensitive
    # endpoint until the cooldown passes (or an admin clears it).
    if path.startswith("/api/") and path not in _LOCKDOWN_ALLOWED_PATHS and security_monitor.is_locked_down():
        st = security_monitor.status()
        return JSONResponse(status_code=503, content={
            "status": "locked",
            "message": "🛡️ ระบบตรวจพบภัยคุกคามด้านความปลอดภัย — ปิดระบบหลังบ้านชั่วคราวเพื่อป้องกันข้อมูล กรุณารอสักครู่ครับ",
            "security": st,
        })

    response = await call_next(request)

    # Defensive HTTP security headers (clickjacking, MIME-sniffing, referrer leak).
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    message: str
    weather: str = ""
    session_id: str = "default"


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "company": settings.company_name,
        "assistant": settings.assistant_name,
        "active_ai": settings.active_ai,
        "knowledge": knowledge.info(),
        "command_execution": settings.enable_command_execution,
        "providers_configured": {
            "claude": bool(settings.claude_api_key),
            "gemini": bool(settings.gemini_api_key),
            "ollama": True,
            "groq": bool(settings.groq_api_key),
        },
        "models": {
            "claude": settings.claude_model,
            "gemini": settings.gemini_model,
            "ollama": settings.ollama_model,
            "groq": settings.groq_model,
        },
    }


@app.get("/api/system-status")
def get_system_status():
    return monitor.snapshot()


@app.get("/api/system-sensors")
def get_system_sensors():
    return monitor.sensors()


@app.get("/api/system-log")
def get_system_log():
    """Most-recent-first feed of real backend log lines (admin logins, commands
    run, knowledge reloads...) for the frontend's SYSTEM LOG panel."""
    return {"entries": list(_log_buffer)[::-1][:20]}


@app.get("/api/token-status")
def get_token_status_endpoint():
    """Per-provider token/request usage today, for the frontend's TOKEN
    STATUS panel — real counters accumulated from each API response's usage
    field, not an estimate."""
    return get_token_status()


@app.get("/api/models")
def list_models_endpoint():
    """JSON list of providers (configured status, model label) + which one
    is active — powers the model-switch dropdown in the AI ENGINE panel."""
    models = [
        {
            "name": name,
            "label": _provider_model_label(name),
            "configured": _provider_configured(name),
            "active": settings.active_ai == name,
        }
        for name in _MODEL_DISPLAY_ORDER
    ]
    return {"active_ai": settings.active_ai, "models": models}


class ModelSwitchRequest(BaseModel):
    session_id: str = "default"
    model: str = ""


@app.post("/api/models/switch")
def switch_model_endpoint(request: ModelSwitchRequest):
    """Switch the active AI model from the web UI — same admin-only,
    global-effect switch as the `[เปลี่ยนโมเดล: ...]` chat command, just
    reachable from a dropdown instead of typed text."""
    if not store.is_admin(request.session_id or "default"):
        return {"status": "error", "message": "🚫 การเปลี่ยนโมเดล AI สงวนไว้สำหรับผู้ดูแลระบบครับ"}
    reply = switch_active_model(request.model)
    return {"status": "ok", "message": reply, "active_ai": settings.active_ai}


class SecurityRequest(BaseModel):
    session_id: str = "default"
    override_code: str = ""  # Superadmin break-glass code — works even without a prior admin session


@app.get("/api/security/status")
def security_status():
    """Live security posture for the frontend shield indicator. Always reachable,
    even during lockdown, so the UI can show the state and poll for recovery."""
    return security_monitor.status()


@app.post("/api/security/clear")
def security_clear(request: SecurityRequest, http_request: Request):
    """Lift an active lockdown early. Two ways in: (1) an already-authenticated
    admin session (from before the lockdown — logins are blocked WHILE locked,
    by design), or (2) the separate Superadmin SECURITY_OVERRIDE_CODE, which
    works even with no prior session — the break-glass path for when nobody
    was logged in when the lockdown engaged."""
    ip = client_ip(http_request)
    is_admin_session = store.is_admin(request.session_id or "default")

    override_ok = False
    if request.override_code:
        if not security_override_rate_limiter.check(ip):
            wait = security_override_rate_limiter.retry_after(ip)
            log.warning("Security override code rate-limited (ip=%s, retry in %ds)", ip, wait)
            return {"status": "error", "message": f"ลองรหัส Security Code บ่อยเกินไปครับ กรุณารออีก {wait} วินาที"}
        override_ok = bool(settings.security_override_code) and hmac.compare_digest(
            request.override_code.encode("utf-8"), settings.security_override_code.encode("utf-8")
        )
        if not override_ok:
            log.warning("Security override code REJECTED (ip=%s)", ip)

    if not (is_admin_session or override_ok):
        return {"status": "error", "message": "🚫 ต้องเป็นแอดมินที่ยืนยันตัวตนแล้ว หรือใส่ Security Code (Superadmin) ให้ถูกต้องครับ"}

    security_monitor.clear()
    log.warning("Security lockdown cleared (session=%s..., via_override_code=%s, ip=%s)",
                (request.session_id or "default")[:8], override_ok, ip)
    return {"status": "ok", "message": "✅ ยกเลิกการล็อกดาวน์และรีเซ็ตประวัติภัยคุกคามแล้วครับ", **security_monitor.status()}


@app.post("/api/security/lockdown")
def security_lockdown(request: SecurityRequest):
    """Admin can manually seal the backend (panic button)."""
    if not store.is_admin(request.session_id or "default"):
        return {"status": "error", "message": "🚫 ต้องเป็นแอดมินที่ยืนยันตัวตนแล้วเท่านั้นครับ"}
    security_monitor.engage_manual("admin panic button")
    return {"status": "ok", "message": "🛑 สั่งล็อกดาวน์ระบบหลังบ้านแล้วครับ", **security_monitor.status()}


@app.get("/api/knowledge")
def get_knowledge():
    return knowledge.info()


@app.post("/api/knowledge/reload")
def reload_knowledge():
    result = knowledge.reload()
    return {"status": "reloaded", **result}


@app.get("/api/training/status")
def training_status():
    """Overview of the AI training pipeline: taught facts, saved logs, dataset."""
    dataset_path = os.path.join(TRAINING_DIR, "dataset_sysnect.jsonl")
    dataset_examples = 0
    if os.path.isfile(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                dataset_examples = sum(1 for line in fh if line.strip())
        except Exception:  # noqa: BLE001
            pass
    return {
        "learned_facts": len(_learned_facts()),
        "learned_file": _learned_path(),
        "log_sessions": len(glob.glob(os.path.join(LOGS_DIR, "chat_log_*.json"))),
        "dataset_examples": dataset_examples,
        "dataset_path": dataset_path,
    }


# --------------------------------------------------------------------------- #
#  Admin Control Panel API (all admin-gated — used by the web admin panel)
# --------------------------------------------------------------------------- #
class TeachRequest(BaseModel):
    fact: str = ""
    session_id: str = "default"


class ForgetRequest(BaseModel):
    index: int = -1
    session_id: str = "default"
    password: str = ""  # re-entered admin password — required even though the session is already admin


class AdminRequest(BaseModel):
    session_id: str = "default"


def _require_admin(session_id: str) -> Optional[dict]:
    """Returns an error dict if the session is not admin, else None."""
    if not store.is_admin(session_id or "default"):
        return {"status": "error", "message": "🚫 ต้องเข้าสู่ระบบแอดมินก่อนครับ"}
    return None


@app.get("/api/knowledge/learned")
def list_learned(session_id: str = "default"):
    """List taught facts (admin only). Returns index + text for each."""
    err = _require_admin(session_id)
    if err:
        return err
    facts = _learned_facts()
    items = [{"index": i, "text": f} for i, f in enumerate(facts)]
    return {"status": "ok", "count": len(items), "facts": items, "kb": knowledge.info()}


@app.post("/api/knowledge/teach")
def api_teach(request: TeachRequest):
    err = _require_admin(request.session_id)
    if err:
        return err
    fact = " ".join((request.fact or "").split())
    if not fact:
        return {"status": "error", "message": "กรุณาพิมพ์ความรู้ที่จะสอนครับ"}
    info = teach_fact(fact)
    return {"status": "ok", "message": f"เรียนรู้แล้ว ({info['total']} รายการ)", **info}


@app.post("/api/knowledge/forget")
def api_forget(request: ForgetRequest):
    err = _require_admin(request.session_id)
    if err:
        return err
    # Deleting knowledge is destructive — re-verify the admin password even
    # though this session is already logged in, so a hijacked/careless admin
    # session (or a friend at the keyboard) can't wipe data without the password.
    if not settings.admin_password or not hmac.compare_digest(
        (request.password or "").encode("utf-8"), settings.admin_password.encode("utf-8")
    ):
        log.warning("Knowledge-forget confirmation FAILED (wrong password, session %s...)", request.session_id[:8])
        return {"status": "error", "message": "🚫 รหัสผ่านไม่ถูกต้องครับ ไม่สามารถลบความรู้ได้ กรุณาใส่รหัสผ่านแอดมินให้ถูกต้อง"}
    result = forget_fact(request.index)
    if not result.get("ok"):
        return {"status": "error", "message": "ลบไม่สำเร็จ (ไม่พบรายการ)", **result}
    log.warning("Knowledge fact forgotten (session %s..., index=%d)", request.session_id[:8], request.index)
    return {"status": "ok", "message": "ลบความรู้แล้ว", **result}


@app.post("/api/training/build")
def api_build_dataset(request: AdminRequest):
    err = _require_admin(request.session_id)
    if err:
        return err
    result = build_training_dataset()
    return {"status": "ok", **result}


# --------------------------------------------------------------------------- #
#  Vision & Clipboard (OS-level access — admin-gated, same trust level as CMD)
# --------------------------------------------------------------------------- #
@app.get("/api/os/screenshot")
def os_screenshot(session_id: str = "default"):
    """Capture the boss's screen and return it as base64 PNG. The frontend
    feeds this straight into the existing /api/upload image pipeline, which
    already knows how to run it through Gemini vision — no new AI code needed."""
    err = _require_admin(session_id)
    if err:
        return err
    try:
        import pyautogui

        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        log.info("Screenshot captured (session %s...)", session_id[:8])
        return {"status": "success", "image": encoded, "mime_type": "image/png"}
    except Exception as exc:  # noqa: BLE001
        log.error("Screenshot failed: %s", exc)
        return {"status": "error", "message": f"ถ่ายภาพหน้าจอไม่สำเร็จครับ: {exc}"}


@app.get("/api/os/clipboard")
def os_clipboard(session_id: str = "default"):
    """Return the current text on the boss's clipboard."""
    err = _require_admin(session_id)
    if err:
        return err
    try:
        import pyperclip

        text = pyperclip.paste() or ""
        log.info("Clipboard read (session %s..., %d chars)", session_id[:8], len(text))
        return {"status": "success", "text": text}
    except Exception as exc:  # noqa: BLE001
        log.error("Clipboard read failed: %s", exc)
        return {"status": "error", "message": f"อ่านคลิปบอร์ดไม่สำเร็จครับ: {exc}"}


# --------------------------------------------------------------------------- #
#  Authentication (login modal — triggered by typing 'login' in the chat UI)
# --------------------------------------------------------------------------- #
class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""
    session_id: str = "default"


@app.post("/api/auth/login")
def auth_login(request: LoginRequest, http_request: Request):
    session_id = request.session_id or "default"
    ip = client_ip(http_request)
    if not login_rate_limiter.check(ip):
        wait = login_rate_limiter.retry_after(ip)
        log.warning("Login rate-limited (ip=%s, retry in %ds)", ip, wait)
        security_monitor.record("rate_limit", ip, "login")
        return {"status": "error", "message": f"พยายามเข้าสู่ระบบบ่อยเกินไปครับ กรุณารออีก {wait} วินาทีแล้วลองใหม่"}
    if not settings.admin_password:
        return {"status": "error", "message": "ยังไม่ได้ตั้งค่า ADMIN_PASSWORD ใน backend/.env ครับ"}
    # timing-safe comparison (bytes — supports non-ASCII passwords)
    user_ok = hmac.compare_digest(
        request.username.strip().lower().encode("utf-8"),
        settings.admin_username.strip().lower().encode("utf-8"),
    )
    pass_ok = hmac.compare_digest(
        request.password.encode("utf-8"),
        settings.admin_password.encode("utf-8"),
    )
    if not (user_ok and pass_ok):
        log.warning("Failed admin login attempt (session %s..., ip=%s)", session_id[:8], ip)
        security_monitor.record("failed_login", ip)
        return {"status": "error", "message": "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้องครับ"}
    store.set_admin(session_id, True)
    log.info("Admin login success (session %s...)", session_id[:8])
    return {
        "status": "success",
        "message": "🔓 **ADMIN MODE UNLOCKED**\nยืนยันตัวตนสำเร็จครับบอส ระบบปลดล็อกสิทธิ์ระดับสูงสุด (God Mode) เรียบร้อย — เข้าถึงคำสั่งระบบ, สอนความรู้ `[สอน]`, สร้างชุดฝึก `[สร้างชุดฝึก]`, ดูรายการโมเดล `[รายการโมเดล]`, และเปลี่ยนโมเดล `[เปลี่ยนโมเดล: ชื่อโมเดล]` ได้เต็มรูปแบบครับ!",
    }


@app.post("/api/auth/logout")
def auth_logout(request: LoginRequest):
    session_id = request.session_id or "default"
    store.set_admin(session_id, False)
    log.info("Admin logout (session %s...)", session_id[:8])
    return {"status": "success", "message": "🔒 **ADMIN MODE LOCKED**\nล็อกสิทธิ์ผู้ดูแลระบบเรียบร้อยครับ กลับสู่โหมดพนักงานปกติ"}


@app.get("/api/auth/status")
def auth_status(session_id: str = "default"):
    return {"is_admin": store.is_admin(session_id)}


def check_connections() -> str:
    report = []
    report.append("⚙️ **ระบบตรวจสอบการเชื่อมต่อ (Connection Status Report)**\n")
    
    # 1. Internet Connection
    try:
        import socket
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        report.append("🌐 **อินเทอร์เน็ต (Internet):** 🟢 เชื่อมต่อปกติ (Connected)")
    except Exception:
        report.append("🌐 **อินเทอร์เน็ต (Internet):** 🔴 ไม่สามารถเชื่อมต่อได้ (Offline)")
        
    # 2. Local Ollama Status
    ollama_running = False
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", 11434))
        s.close()
        ollama_running = True
        report.append("🦙 **บริการ Ollama (Local):** 🟢 เปิดอยู่ (Running on port 11434)")
    except Exception:
        report.append("🦙 **บริการ Ollama (Local):** 🔴 ปิดอยู่ (Not responding on port 11434)")

    # 3. Active AI Status
    provider = settings.active_ai.lower()
    report.append(f"🤖 **สมองกลหลัก (ACTIVE_AI):** `{provider.upper()}`")
    
    if provider == "ollama":
        if ollama_running:
            try:
                import urllib.request
                import json
                req = urllib.request.Request(f"{settings.ollama_base_url.replace('/v1', '')}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    models = [m["name"] for m in data.get("models", [])]
                if settings.ollama_model in models or f"{settings.ollama_model}:latest" in models:
                    report.append(f"   └─ 🟢 โมเดล `{settings.ollama_model}`: พร้อมใช้งาน (Ready)")
                else:
                    report.append(f"   └─ ⚠️ โมเดล `{settings.ollama_model}`: ไม่พบใน Ollama (โมเดลที่มี: {', '.join(models)})")
            except Exception as e:
                report.append(f"   └─ 🔴 ทดสอบเชื่อมต่อ API ล้มเหลว: {e}")
        else:
            report.append(f"   └─ 🔴 ไม่สามารถตรวจสอบโมเดลได้เนื่องจากบริการ Ollama ปิดอยู่")
            
    elif provider in ("gemini", "google"):
        if not settings.gemini_api_key:
            report.append("   └─ 🔴 ยังไม่ได้ตั้งค่า GEMINI_API_KEY ในไฟล์ .env")
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel(settings.gemini_model)
                model.count_tokens("test")
                report.append(f"   └─ 🟢 คีย์ API และการเชื่อมต่อ Gemini `{settings.gemini_model}`: ปกติ (Ready)")
            except Exception as e:
                report.append(f"   └─ 🔴 เชื่อมต่อ API Gemini ล้มเหลว: {e}")
                
    elif provider in ("claude", "anthropic"):
        if not settings.claude_api_key:
            report.append("   └─ 🔴 ยังไม่ได้ตั้งค่า CLAUDE_API_KEY ในไฟล์ .env")
        else:
            report.append(f"   └─ 🟢 คีย์ API Claude ตั้งค่าไว้แล้ว (โมเดล: `{settings.claude_model}`)")
            
    elif provider == "groq":
        if not settings.groq_api_key:
            report.append("   └─ 🔴 ยังไม่ได้ตั้งค่า GROQ_API_KEY ในไฟล์ .env")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
                client.models.list()
                report.append(f"   └─ 🟢 คีย์ API และการเชื่อมต่อ Groq `{settings.groq_model}`: ปกติ (Ready)")
            except Exception as e:
                report.append(f"   └─ 🔴 เชื่อมต่อ API Groq ล้มเหลว: {e}")
                
    # 4. System Info
    import psutil
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    report.append(f"\n💻 **ทรัพยากรเครื่อง:** CPU: {cpu}% | RAM: {mem}%")
    
    return "\n".join(report)


@app.get("/api/chat/history")
def get_chat_history(session_id: str = "default"):
    return {"history": store.get_history(session_id)}


def get_auto_routed_provider(user_msg: str) -> str:
    # Check key configurations
    groq_ok = bool(settings.groq_api_key)
    gemini_ok = bool(settings.gemini_api_key)
    
    # Check if Ollama is listening locally (200ms timeout)
    ollama_ok = False
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(("127.0.0.1", 11434))
        s.close()
        ollama_ok = True
    except Exception:
        pass
        
    low = user_msg.lower()
    
    # 1. Vision / Image related
    if any(k in low for k in ("รูป", "ภาพ", "image", "photo", "png", "jpg", "screenshot")):
        if gemini_ok:
            return "gemini"
            
    # 2. Coding / Technical / Mathematics
    coding_keywords = (
        "code", "เขียนโค้ด", "เขียนโปรแกรม", "script", "python", "html", 
        "css", "javascript", "ตาราง", "บั๊ก", "error", "คำนวณ", "เลข", "สูตร"
    )
    if any(k in low for k in coding_keywords):
        if groq_ok:
            return "groq"
        if gemini_ok:
            return "gemini"
            
    # 3. System command execution (e.g. open notepad, run shell)
    system_keywords = (
        "เปิดโปรแกรม", "เปิดแอพ", "เปิดเว็บ", "cmd", "powershell", 
        "รันคำสั่ง", "clean", "ลบไฟล์", "เช็คสถานะ"
    )
    if any(k in low for k in system_keywords):
        if groq_ok:
            return "groq"
        if ollama_ok:
            return "ollama"
            
    # Default routing chain
    if groq_ok:
        return "groq"
    if gemini_ok:
        return "gemini"
    if ollama_ok:
        return "ollama"
        
    return settings.active_ai # fallback


@app.post("/api/chat")
def chat_with_jarvis(request: ChatRequest, http_request: Request):
    session_id = request.session_id or "default"
    user_msg = (request.message or "")[:MAX_MESSAGE_CHARS]

    ip = client_ip(http_request)
    if any(bad in (request.session_id or "") for bad in ("..", "/", "\\", "\x00")):
        security_monitor.record("path_traversal", ip, request.session_id[:40])
    if not chat_rate_limiter.check(ip):
        wait = chat_rate_limiter.retry_after(ip)
        log.warning("Chat rate-limited (ip=%s, retry in %ds)", ip, wait)
        security_monitor.record("rate_limit", ip, "chat")
        return {"reply": f"ส่งข้อความถี่เกินไปครับ กรุณารออีก {wait} วินาทีแล้วลองใหม่ครับ", "provider": settings.active_ai}

    # Resolve dynamic AI provider if AUTO routing is selected
    provider = settings.active_ai
    if provider == "auto":
        provider = get_auto_routed_provider(user_msg)
        log.info("[AUTO ROUTER] Routed request to: %s", provider.upper())

    log.info("Chat request (session %s..., provider=%s, ip=%s): %s", session_id[:8], provider, ip, user_msg[:200])

    # Re-confirming a held destructive action (e.g. "[ยืนยันลบ: password]") takes
    # priority over everything else — never send this to the AI model.
    confirm_reply = handle_confirm_destructive(session_id, user_msg)
    if confirm_reply is not None:
        store.add_message(session_id, "user", user_msg)
        store.add_message(session_id, "assistant", confirm_reply)
        return {"reply": confirm_reply, "provider": provider}

    # Check Connection Command
    if user_msg.strip().lower() in ("check connection", "เช็คการเชื่อมต่อ", "checkconnection"):
        reply = check_connections()
        return {"reply": reply, "provider": provider}

    if "[save log data]" in user_msg.lower():
        import json
        import os
        log_dir = LOGS_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"chat_log_{safe_session_key(session_id)}.json")
        history = store.get_history(session_id)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return {"reply": f"ระบบได้ทำการบันทึก Log ข้อมูลการสนทนาลงใน `{log_path}` เรียบร้อยแล้วครับบอส", "provider": provider}

    # --- AI model switching (admin-only — it's a GLOBAL setting, affects everyone) ---
    if user_msg.strip().lower() in _LIST_MODELS_TEXTS:
        return {"reply": list_models_info(), "provider": provider}

    switch_match = _SWITCH_MODEL_RE.match(user_msg)
    if switch_match:
        if not store.is_admin(session_id):
            return {"reply": "🚫 การเปลี่ยนโมเดล AI สงวนไว้สำหรับผู้ดูแลระบบครับ (มีผลกับทุกคนที่ใช้ระบบ) กรุณาใช้คำสั่งยืนยันตัวตนก่อนครับบอส", "provider": provider}
        reply = switch_active_model(switch_match.group(1))
        return {"reply": reply, "provider": settings.active_ai}

    # --- AI Training commands ------------------------------------------- #
    fact = _extract_teach_fact(user_msg)
    if fact is not None:
        if not store.is_admin(session_id):
            return {"reply": "🚫 การสอนความรู้ถาวรสงวนไว้สำหรับผู้ดูแลระบบครับ กรุณาใช้คำสั่งยืนยันตัวตน (Authentication) ก่อนครับบอส", "provider": provider}
        if not fact:
            return {"reply": "พิมพ์ `[สอน]` ตามด้วยความรู้ที่ต้องการให้ผมจำถาวรครับบอส เช่น:\n`[สอน] SYSNECT มีพนักงานทั้งหมด 12 คน`", "provider": provider}
        info = teach_fact(fact)
        return {"reply": f"🧠 **เรียนรู้แล้วครับบอส!**\nผมบันทึกความรู้ใหม่ลงสมองถาวร (`knowledge/learned.md`) และโหลดเข้าระบบทันที:\n\n> {fact}\n\n📚 ความรู้ที่ถูกสอนสะสมทั้งหมด: **{info['total']} รายการ** — พร้อมใช้ตอบทุกคนตั้งแต่ข้อความถัดไปครับ", "provider": provider}

    stripped = user_msg.strip().lower()
    if stripped in ("[ความรู้]", "[learned]"):
        facts = _learned_facts()
        if not facts:
            return {"reply": "ยังไม่มีความรู้ที่ถูกสอนเพิ่มครับบอส — ปลดล็อกแอดมินแล้วใช้ `[สอน] <ความรู้>` เพื่อเริ่มสอนผมได้เลยครับ 🧠", "provider": provider}
        recent = "\n".join(facts[-10:])
        return {"reply": f"📚 **ความรู้ที่ผมถูกสอนเพิ่ม ({len(facts)} รายการ)** — แสดง 10 รายการล่าสุด:\n\n{recent}\n\n(ไฟล์เต็มอยู่ที่ `knowledge/learned.md`)", "provider": provider}

    if stripped in ("[สร้างชุดฝึก]", "[build dataset]"):
        if not store.is_admin(session_id):
            return {"reply": "🚫 การสร้างชุดข้อมูลฝึกสงวนไว้สำหรับผู้ดูแลระบบครับ กรุณาใช้คำสั่งยืนยันตัวตน (Authentication) ก่อนครับบอส", "provider": provider}
        result = build_training_dataset()
        if result["examples"] == 0:
            return {"reply": "🎓 ยังไม่มีบทสนทนาให้สร้างชุดฝึกครับบอส\n\n**วิธีสะสมข้อมูลฝึก:**\n1. คุยกับผมตามปกติ (ยิ่งเยอะยิ่งดี)\n2. จบแต่ละรอบ พิมพ์ `[save log data]` เพื่อบันทึกบทสนทนา\n3. กลับมาสั่ง `[สร้างชุดฝึก]` อีกครั้งครับ", "provider": provider}
        return {"reply": f"🎓 **สร้างชุดข้อมูลฝึกสำเร็จครับบอส!**\n- ตัวอย่างการสนทนา: **{result['examples']} คู่** (จาก {result['log_files']} ไฟล์ log)\n- บันทึกที่: `{result['path']}`\n\n**ขั้นตอน fine-tune จริง (LoRA บน GPU):**\n1. เปิด `finetune.py` แล้วชี้ `data_files` มาที่ไฟล์นี้\n2. รันในสภาพแวดล้อมที่มี Unsloth (ดู `training/README.md`)\n3. ได้ไฟล์ GGUF → import เข้า Ollama → ตั้ง `ACTIVE_AI=ollama`", "provider": provider}

    store.add_message(session_id, "user", user_msg)
    is_admin = store.is_admin(session_id)
    # In admin mode, drop stale "OS locked" refusals so the model can't copy them.
    history = _history_for_model(store.get_history(session_id), is_admin)
    system = build_system_prompt(request.weather, store.get_document(session_id), is_admin)

    reply = generate_reply(provider, system, history)

    # Check for SEARCH command
    search_match = re.search(r"\[SEARCH:\s*(.*?)\]", reply, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip()
        log.info("Web search triggered (session %s...): %s", session_id[:8], query)
        search_results = perform_web_search(query)
        temp_history = history + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"ผลการค้นหาสำหรับ '{query}':\n{search_results}\n\nกรุณาตอบคำถามโดยใช้ข้อมูลจากผลการค้นหานี้"}
        ]
        second_reply = generate_reply(provider, system, temp_history)
        # Drop the raw [SEARCH: ...] tag so the user never sees it in the reply.
        first_clean = re.sub(r"\[SEARCH:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
        reply = f"{first_clean}\n\n{second_reply}".strip() if first_clean else second_reply

    # Check for SYSNECT_DATA command (local search over real SYSNECT project files)
    # ADMIN-ONLY: this reads the company's local source/config, so a non-admin
    # (or prompt-injected) tag is stripped and never runs.
    sysnect_match = re.search(r"\[SYSNECT_DATA:\s*(.*?)\]", reply, re.IGNORECASE)
    if sysnect_match and not is_admin:
        log.warning("Blocked SYSNECT_DATA search from a NON-ADMIN session %s... (possible injection)", session_id[:8])
        security_monitor.record("nonadmin_sysnect", ip)
        reply = re.sub(r"\[SYSNECT_DATA:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
    elif sysnect_match:
        query = sysnect_match.group(1).strip()
        log.info("SYSNECT data search triggered (session %s...): %s", session_id[:8], query)
        sysnect_results = search_sysnect_data(query)
        temp_history = history + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"ผลการค้นข้อมูล SYSNECT สำหรับ '{query}':\n{sysnect_results}\n\nกรุณาตอบคำถามโดยใช้ข้อมูลจากผลการค้นนี้เท่านั้น"}
        ]
        second_reply = generate_reply(provider, system, temp_history)
        first_clean = re.sub(r"\[SYSNECT_DATA:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
        reply = f"{first_clean}\n\n{second_reply}".strip() if first_clean else second_reply

    reply = agent.process(reply, session_id, is_admin)

    if not reply:
        reply = fallback_reply(user_msg)

    store.add_message(session_id, "assistant", reply)
    log.info("Chat reply sent (session %s..., provider=%s, %d chars)", session_id[:8], provider, len(reply))
    return {"reply": reply, "provider": provider}


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    """Streaming chat (Server-Sent Events) for a modern typewriter UX.
    Emits {"delta": "..."} chunks, then a final {"done": true, "reply": "..."}.
    """
    session_id = request.session_id or "default"
    user_msg = (request.message or "")[:MAX_MESSAGE_CHARS]

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    ip = client_ip(http_request)
    if not chat_rate_limiter.check(ip):
        wait = chat_rate_limiter.retry_after(ip)
        log.warning("Chat stream rate-limited (ip=%s, retry in %ds)", ip, wait)
        security_monitor.record("rate_limit", ip, "chat_stream")

        def limited():
            yield sse({"done": True, "reply": f"ส่งข้อความถี่เกินไปครับ กรุณารออีก {wait} วินาทีแล้วลองใหม่ครับ", "provider": settings.active_ai})

        return StreamingResponse(limited(), media_type="text/event-stream")

    # Resolve dynamic AI provider if AUTO routing is selected
    provider = settings.active_ai
    if provider == "auto":
        provider = get_auto_routed_provider(user_msg)
        log.info("[AUTO ROUTER] Routed stream to: %s", provider.upper())

    log.info("Chat stream request (session %s..., provider=%s, ip=%s): %s", session_id[:8], provider, ip, user_msg[:200])

    # Reuse the non-streaming handler for special admin/command messages.
    _stripped = user_msg.strip().lower()
    if (
        "[save log data]" in user_msg.lower()
        or _extract_teach_fact(user_msg) is not None
        or _stripped in ("[ความรู้]", "[learned]", "[สร้างชุดฝึก]", "[build dataset]")
        or _stripped in ("check connection", "เช็คการเชื่อมต่อ", "checkconnection")
        or _CONFIRM_DESTRUCTIVE_RE.match(user_msg) is not None
        or _stripped in _LIST_MODELS_TEXTS
        or _SWITCH_MODEL_RE.match(user_msg) is not None
    ):
        result = chat_with_jarvis(request, http_request)

        def one():
            yield sse({"done": True, "reply": result["reply"], "provider": result["provider"]})

        return StreamingResponse(one(), media_type="text/event-stream")

    store.add_message(session_id, "user", user_msg)
    is_admin = store.is_admin(session_id)
    # In admin mode, drop stale "OS locked" refusals so the model can't copy them.
    history = _history_for_model(store.get_history(session_id), is_admin)
    system = build_system_prompt(request.weather, store.get_document(session_id), is_admin)

    def gen():
        parts: List[str] = []
        for chunk in generate_reply_stream(provider, system, history):
            parts.append(chunk)
            yield sse({"delta": chunk})
            
        reply = "".join(parts)
        
        # Check for SEARCH command
        search_match = re.search(r"\[SEARCH:\s*(.*?)\]", reply, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            log.info("Web search triggered (session %s..., stream): %s", session_id[:8], query)
            yield sse({"delta": f"\n\n*กำลังค้นหาข้อมูล: {query}...*\n\n"})
            search_results = perform_web_search(query)
            
            temp_history = history + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"ผลการค้นหาสำหรับ '{query}':\n{search_results}\n\nกรุณาตอบคำถามโดยใช้ข้อมูลจากผลการค้นหานี้"}
            ]
            
            parts2 = []
            for chunk in generate_reply_stream(provider, system, temp_history):
                parts2.append(chunk)
                yield sse({"delta": chunk})

            # Drop the raw [SEARCH: ...] tag from the stored/final reply.
            first_clean = re.sub(r"\[SEARCH:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
            reply = f"{first_clean}\n\n{''.join(parts2)}".strip() if first_clean else "".join(parts2)

        # Check for SYSNECT_DATA command (local search over real SYSNECT project files)
        # ADMIN-ONLY (reads local source/config) — non-admin/injected tag is stripped.
        sysnect_match = re.search(r"\[SYSNECT_DATA:\s*(.*?)\]", reply, re.IGNORECASE)
        if sysnect_match and not is_admin:
            log.warning("Blocked SYSNECT_DATA search from a NON-ADMIN session %s... (stream, possible injection)", session_id[:8])
            security_monitor.record("nonadmin_sysnect", ip)
            reply = re.sub(r"\[SYSNECT_DATA:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
        elif sysnect_match:
            query = sysnect_match.group(1).strip()
            log.info("SYSNECT data search triggered (session %s..., stream): %s", session_id[:8], query)
            yield sse({"delta": f"\n\n*กำลังค้นข้อมูล SYSNECT: {query}...*\n\n"})
            sysnect_results = search_sysnect_data(query)

            temp_history = history + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"ผลการค้นข้อมูล SYSNECT สำหรับ '{query}':\n{sysnect_results}\n\nกรุณาตอบคำถามโดยใช้ข้อมูลจากผลการค้นนี้เท่านั้น"}
            ]

            parts3 = []
            for chunk in generate_reply_stream(provider, system, temp_history):
                parts3.append(chunk)
                yield sse({"delta": chunk})

            # Drop the raw [SYSNECT_DATA: ...] tag from the stored/final reply.
            first_clean = re.sub(r"\[SYSNECT_DATA:\s*.*?\]", "", reply, flags=re.IGNORECASE).strip()
            reply = f"{first_clean}\n\n{''.join(parts3)}".strip() if first_clean else "".join(parts3)

        cleaned = agent.process(reply, session_id, is_admin)
        if not cleaned:
            cleaned = fallback_reply(user_msg)
        store.add_message(session_id, "assistant", cleaned)
        log.info("Chat stream reply sent (session %s..., provider=%s, %d chars)", session_id[:8], provider, len(cleaned))
        yield sse({"done": True, "reply": cleaned, "provider": provider})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = "default"):
    try:
        content = await file.read()
        # Defense-in-depth size cap (the request middleware also guards Content-Length).
        if len(content) > MAX_REQUEST_BYTES:
            log.warning("Rejected oversized upload: %s (%d bytes)", file.filename, len(content))
            return {"status": "error", "message": "ไฟล์มีขนาดใหญ่เกินไปครับ"}
        filename = (file.filename or "").lower()
        text = ""
        is_image = filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))
        log.info("File uploaded (session %s...): %s (%d bytes, image=%s)",
                 session_id[:8], file.filename, len(content), is_image)

        if filename.endswith(".pdf"):
            import PyPDF2

            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif filename.endswith(".docx"):
            import docx

            document = docx.Document(io.BytesIO(content))
            text = "\n".join(para.text for para in document.paragraphs)
        elif is_image:
            if settings.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                image_data = {
                    "mime_type": file.content_type or "image/png",
                    "data": content
                }
                prompt = (
                    "คุณคือส่วนวิเคราะห์และอ่านรูปภาพของ JARVIS\n"
                    "วิเคราะห์รูปภาพนี้แล้วสกัดข้อความทั้งหมดออกมาอย่างละเอียด หากเป็นรูปที่มีข้อความภาษาไทย/อังกฤษ ให้ทำ OCR ถอดข้อความออกมาทั้งหมดโดยรักษาการจัดหน้า/ย่อหน้าไว้ให้ใกล้เคียงที่สุด\n"
                    "หากเป็นรูปหน้าจอโปรแกรม แผนภูมิ โค้ดโปรแกรม หรือตาราง ให้พยายามอธิบายและแปลงรายละเอียดเหล่านั้นออกมาเป็นข้อความและโค้ดให้ละเอียดถี่ถ้วน\n"
                    "หากเป็นรูปภาพทั่วไป ให้เขียนคำอธิบายรายละเอียดที่เกิดขึ้นในรูปภาพอย่างกระชับและเข้าใจง่าย และตอบกลับมาเป็นภาษาไทยเป็นหลัก"
                )
                response = model.generate_content([prompt, image_data])
                text = response.text
            else:
                text = "[ข้อผิดพลาด: ไม่สามารถวิเคราะห์รูปภาพได้เนื่องจากไม่ได้ตั้งค่า GEMINI_API_KEY ใน backend/.env]"
        else:
            text = content.decode("utf-8", errors="ignore")

        text = text[: settings.doc_context_limit]
        store.set_document(session_id, text)
        
        if is_image:
            msg = f"วิเคราะห์รูปภาพ {file.filename} เรียบร้อยแล้วครับท่าน รายละเอียดดังนี้:\n\n{text[:500]}..."
            user_msg = f"[อัปโหลดรูปภาพ: {file.filename}]"
        else:
            msg = f"อ่านไฟล์ {file.filename} เรียบร้อยแล้วครับท่าน ความยาว {len(text)} ตัวอักษร"
            user_msg = f"[อัปโหลดไฟล์: {file.filename}]"

        # Save to chat history so it persists across reloads
        store.add_message(session_id, "user", user_msg)
        store.add_message(session_id, "assistant", msg)

        return {
            "status": "success",
            "message": msg,
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Upload error: %s", exc)
        return {"status": "error", "message": f"เกิดข้อผิดพลาดในการอ่านไฟล์: {exc}"}


if __name__ == "__main__":
    log.info("[JARVIS] Backend starting on port 8000 (active AI: %s)", settings.active_ai)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
